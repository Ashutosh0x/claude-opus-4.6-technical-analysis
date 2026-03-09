"""
Continuous Batching & PagedAttention Scheduler.

The core inference serving optimization for high-throughput LLM serving.

Without continuous batching:
    Step 1: [A-tok1, B-tok45, C-tok200, _, _, _, _]  ← slots wasted
    Step 2: [A-tok2, B-tok46, C-DONE,   _, _, _, _]  ← C done, slot empty

With continuous batching:
    Step 1: [A-tok1, B-tok45, C-tok200]
    Step 2: [A-tok2, B-tok46, D-tok1(new!)]  ← D fills C's slot immediately

GPU utilization: ~2–3× improvement over static batching.

PagedAttention (vLLM, Kwon et al. 2023):
    Inspired by OS virtual memory paging:
    - KV cache divided into fixed-size pages (blocks of B tokens)
    - Each sequence mapped to non-contiguous pages via page table
    - Pages allocated on demand, freed when sequence completes
    - Memory utilization > 95% (vs ~50% with pre-allocation)

    N_pages = ceil(S_current / B_block)

Prefix caching with PagedAttention:
    Shared system prompts map to SAME physical pages across requests:
    Memory(N requests) = M_shared_prefix + N × M_unique_suffix
    instead of N × (M_prefix + M_suffix)

    For 1000 concurrent requests with 10K-token system prompt,
    saves ~12+ TB of KV cache.

References:
    - vLLM: Kwon et al. 2023 (arXiv:2309.06180)
    - Orca: Yu et al. 2022 (iteration-level scheduling)
    - Sarathi: Agrawal et al. 2024 (chunked prefill)
"""

import time
import heapq
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request State Machine
# ---------------------------------------------------------------------------

class RequestState(Enum):
    WAITING    = "waiting"     # in queue, not yet started
    PREFILL    = "prefill"     # processing input prompt
    DECODE     = "decode"      # generating output tokens
    COMPLETE   = "complete"    # finished (EOS or max_tokens)
    CANCELLED  = "cancelled"   # cancelled by client
    ERROR      = "error"       # failed


@dataclass
class InferenceRequest:
    """
    A single API request with its full lifecycle state.

    Lifecycle:
        WAITING → PREFILL → DECODE → COMPLETE
                                   → ERROR (on failure)
                         → CANCELLED (client disconnect)
    """
    request_id: str
    input_ids: List[int]
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = True

    # State
    state: RequestState = RequestState.WAITING
    output_ids: List[int] = field(default_factory=list)
    kv_cache_pages: List[int] = field(default_factory=list)  # page indices
    created_at: float = field(default_factory=time.time)
    first_token_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Thinking mode
    thinking_tokens: int = 0
    thinking_budget: int = 0

    # Priority (lower = higher priority)
    priority: int = 0

    # Callback for streaming
    on_token: Optional[Callable] = None

    @property
    def total_tokens(self) -> int:
        return len(self.input_ids) + len(self.output_ids)

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_at and self.created_at:
            return (self.first_token_at - self.created_at) * 1000
        return None

    @property
    def total_ms(self) -> Optional[float]:
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at) * 1000
        return None


# ---------------------------------------------------------------------------
# Page Table — Virtual Memory for KV Cache
# ---------------------------------------------------------------------------

@dataclass
class KVPage:
    """
    One page (block) of KV cache.

    Each page holds B_block tokens worth of key-value tensors for
    all layers and KV heads:

    Memory per page:
        M = 2 × L × n_kv × d_h × B_block × dtype_bytes
        = 2 × 160 × 16 × 128 × 256 × 2 = ~320 MB per page
    """
    page_id: int
    block_size: int = 256   # tokens per page
    ref_count: int = 1      # for shared prefix pages
    tokens_used: int = 0

    @property
    def is_full(self) -> bool:
        return self.tokens_used >= self.block_size

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class PageTable:
    """
    Virtual memory page table for KV cache management.

    Maps request sequences to non-contiguous pages of KV cache,
    enabling:
        1. Dynamic allocation (no pre-allocated max_seq_len)
        2. Shared prefix pages (system prompts cached once)
        3. Immediate deallocation on request completion

    Memory utilization > 95% (vs ~50% with static allocation).
    Internal fragmentation < 4% (only last page partially used).
    """

    def __init__(
        self,
        total_pages: int = 10000,
        block_size: int = 256,
    ):
        self.block_size = block_size
        self.total_pages = total_pages

        # Free page pool
        self.free_pages: List[int] = list(range(total_pages))
        self.pages: Dict[int, KVPage] = {
            i: KVPage(page_id=i, block_size=block_size) for i in range(total_pages)
        }

        # Prefix cache: hash(prefix_tokens) → page_ids
        self.prefix_cache: Dict[int, List[int]] = {}
        self.prefix_cache_ttl: Dict[int, float] = {}  # hash → last access time
        self.cache_ttl_seconds: float = 300.0  # 5 minutes

    def allocate(self, num_tokens: int) -> List[int]:
        """Allocate pages for a new sequence of num_tokens."""
        num_pages_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_pages) < num_pages_needed:
            logger.warning(
                f"Not enough free pages: need {num_pages_needed}, "
                f"have {len(self.free_pages)}"
            )
            return []

        allocated = []
        for _ in range(num_pages_needed):
            page_id = self.free_pages.pop()
            self.pages[page_id].ref_count = 1
            self.pages[page_id].tokens_used = min(
                self.block_size,
                num_tokens - len(allocated) * self.block_size
            )
            allocated.append(page_id)

        return allocated

    def free(self, page_ids: List[int]) -> None:
        """Free pages when a sequence completes."""
        for pid in page_ids:
            page = self.pages[pid]
            page.ref_count -= 1
            if page.ref_count <= 0:
                page.ref_count = 0
                page.tokens_used = 0
                self.free_pages.append(pid)

    def share_prefix(
        self, prefix_hash: int, page_ids: List[int]
    ) -> None:
        """Register pages as shared prefix (for prefix caching)."""
        self.prefix_cache[prefix_hash] = page_ids
        self.prefix_cache_ttl[prefix_hash] = time.time()
        for pid in page_ids:
            self.pages[pid].ref_count += 1  # extra ref for cache

    def get_cached_prefix(
        self, prefix_hash: int
    ) -> Optional[List[int]]:
        """Look up cached prefix pages. Returns None on miss."""
        if prefix_hash not in self.prefix_cache:
            return None

        # Check TTL
        last_access = self.prefix_cache_ttl[prefix_hash]
        if time.time() - last_access > self.cache_ttl_seconds:
            # Expired — evict
            self.evict_prefix(prefix_hash)
            return None

        # Refresh TTL on hit
        self.prefix_cache_ttl[prefix_hash] = time.time()
        page_ids = self.prefix_cache[prefix_hash]

        # Increment ref count for new request using this prefix
        for pid in page_ids:
            self.pages[pid].ref_count += 1

        return page_ids

    def evict_prefix(self, prefix_hash: int) -> None:
        """Evict cached prefix, freeing pages if no more references."""
        if prefix_hash not in self.prefix_cache:
            return
        page_ids = self.prefix_cache.pop(prefix_hash)
        self.prefix_cache_ttl.pop(prefix_hash, None)
        self.free(page_ids)

    @property
    def utilization(self) -> float:
        """Current memory utilization (higher = less waste)."""
        used = self.total_pages - len(self.free_pages)
        return used / self.total_pages if self.total_pages > 0 else 0

    @property
    def cache_hit_rate(self) -> float:
        """Placeholder — in production, tracked by request stats."""
        return 0.0


# ---------------------------------------------------------------------------
# Continuous Batching Scheduler
# ---------------------------------------------------------------------------

class ContinuousBatchingScheduler:
    """
    Iteration-level scheduler for continuous batching.

    Each step:
        1. Pop new requests from waiting queue (up to free slots)
        2. Run prefill for new requests (compute-bound, parallel)
        3. Run decode for ongoing requests (memory-bound, one-at-a-time)
        4. Remove completed requests, recycle their pages & slots

    Scheduling policies:
        - FCFS (first come first served): simplest, fair
        - SJF (shortest job first): minimizes avg latency
        - Priority: enterprise > pro > free tier

    Prefill-decode disaggregation (Sarathi):
        Run prefill on separate GPU group from decode for better
        utilization (prefill is compute-bound, decode is memory-bound).
    """

    def __init__(
        self,
        max_batch_size: int = 256,
        max_total_tokens: int = 1_000_000,
        page_table: Optional[PageTable] = None,
        scheduling_policy: str = "fcfs",    # fcfs, sjf, priority
    ):
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.page_table = page_table or PageTable()
        self.scheduling_policy = scheduling_policy

        # Queues
        self.waiting_queue: List[InferenceRequest] = []
        self.running_batch: List[InferenceRequest] = []
        self.completed: List[InferenceRequest] = []

        # Stats
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_prefill_tokens = 0

    def add_request(self, request: InferenceRequest) -> None:
        """Add a new request to the waiting queue."""
        self.total_requests += 1

        # Check for prefix cache hit
        prefix_hash = hash(tuple(request.input_ids[:1024]))
        cached_pages = self.page_table.get_cached_prefix(prefix_hash)
        if cached_pages:
            request.kv_cache_pages = list(cached_pages)
            logger.debug(
                f"Prefix cache HIT for request {request.request_id}: "
                f"{len(cached_pages)} pages"
            )

        self.waiting_queue.append(request)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request (e.g., client disconnect)."""
        for req in self.waiting_queue:
            if req.request_id == request_id:
                req.state = RequestState.CANCELLED
                self.waiting_queue.remove(req)
                return True

        for req in self.running_batch:
            if req.request_id == request_id:
                req.state = RequestState.CANCELLED
                self.page_table.free(req.kv_cache_pages)
                self.running_batch.remove(req)
                return True

        return False

    def _select_new_requests(self) -> List[InferenceRequest]:
        """Select requests from waiting queue to add to batch."""
        available_slots = self.max_batch_size - len(self.running_batch)
        if available_slots <= 0 or not self.waiting_queue:
            return []

        # Sort by scheduling policy
        if self.scheduling_policy == "sjf":
            self.waiting_queue.sort(key=lambda r: r.max_new_tokens)
        elif self.scheduling_policy == "priority":
            self.waiting_queue.sort(key=lambda r: r.priority)
        # else: FCFS (queue order)

        # Check token budget
        current_tokens = sum(r.total_tokens for r in self.running_batch)
        selected = []

        for req in self.waiting_queue[:]:
            if len(selected) >= available_slots:
                break
            if current_tokens + len(req.input_ids) > self.max_total_tokens:
                continue
            # Allocate pages
            if not req.kv_cache_pages:
                pages = self.page_table.allocate(len(req.input_ids))
                if not pages:
                    continue   # not enough memory
                req.kv_cache_pages = pages

            selected.append(req)
            current_tokens += len(req.input_ids)

        for req in selected:
            self.waiting_queue.remove(req)

        return selected

    def step(self, model_forward_fn: Callable) -> Dict[str, Any]:
        """
        Execute one iteration of continuous batching.

        Args:
            model_forward_fn: function that takes batch of requests
                              and returns next tokens + logits

        Returns:
            dict with "new_tokens", "completed", "batch_size", etc.
        """
        t0 = time.time()

        # 1. Admit new requests
        new_requests = self._select_new_requests()
        for req in new_requests:
            req.state = RequestState.PREFILL
            self.running_batch.append(req)

        if not self.running_batch:
            return {"batch_size": 0, "new_tokens": 0, "completed": 0}

        # 2. Run model forward pass (prefill + decode combined)
        # In practice, this runs the model on the entire batch in parallel
        results = model_forward_fn(self.running_batch)

        # 3. Process results
        completed_this_step = []
        new_tokens_count = 0

        for req, result in zip(self.running_batch, results):
            if req.state == RequestState.PREFILL:
                req.state = RequestState.DECODE
                self.total_prefill_tokens += len(req.input_ids)

            token_id = result.get("token_id")
            if token_id is not None:
                req.output_ids.append(token_id)
                new_tokens_count += 1
                self.total_tokens_generated += 1

                if req.first_token_at is None:
                    req.first_token_at = time.time()

                # Stream callback
                if req.on_token:
                    req.on_token(token_id, result.get("text", ""))

                # Check completion
                eos = result.get("is_eos", False)
                max_reached = len(req.output_ids) >= req.max_new_tokens

                if eos or max_reached:
                    req.state = RequestState.COMPLETE
                    req.completed_at = time.time()
                    completed_this_step.append(req)

        # 4. Clean up completed requests
        for req in completed_this_step:
            self.running_batch.remove(req)

            # Cache prefix for future requests
            prefix_hash = hash(tuple(req.input_ids[:1024]))
            if len(req.kv_cache_pages) > 0:
                # Cache the input portion's pages
                input_pages = req.kv_cache_pages[
                    :(len(req.input_ids) + 255) // 256
                ]
                self.page_table.share_prefix(prefix_hash, input_pages)

            # Free output pages
            output_pages = req.kv_cache_pages[
                (len(req.input_ids) + 255) // 256:
            ]
            self.page_table.free(output_pages)

            self.completed.append(req)

        step_ms = (time.time() - t0) * 1000

        return {
            "batch_size": len(self.running_batch),
            "new_requests": len(new_requests),
            "new_tokens": new_tokens_count,
            "completed": len(completed_this_step),
            "step_ms": step_ms,
            "waiting": len(self.waiting_queue),
            "page_utilization": self.page_table.utilization,
            "tokens_per_second": new_tokens_count / (step_ms / 1000) if step_ms > 0 else 0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall serving statistics."""
        completed = [r for r in self.completed if r.state == RequestState.COMPLETE]
        ttfts = [r.ttft_ms for r in completed if r.ttft_ms]
        totals = [r.total_ms for r in completed if r.total_ms]

        return {
            "total_requests": self.total_requests,
            "completed_requests": len(completed),
            "pending_requests": len(self.waiting_queue),
            "active_requests": len(self.running_batch),
            "total_tokens_generated": self.total_tokens_generated,
            "total_prefill_tokens": self.total_prefill_tokens,
            "page_utilization": self.page_table.utilization,
            "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
            "p50_ttft_ms": sorted(ttfts)[len(ttfts) // 2] if ttfts else 0,
            "p99_ttft_ms": sorted(ttfts)[int(len(ttfts) * 0.99)] if ttfts else 0,
            "avg_total_ms": sum(totals) / len(totals) if totals else 0,
        }
