"""
Cache Controller FSM Simulation
================================
Architecture
  - Block size  : 4 words = 16 bytes
  - Cache size  : 16 KB   = 16,384 bytes → 1,024 blocks
  - Address     : 32-bit byte-addressed

Address decomposition (byte address → 32 bits)
  [ TAG (20 bits) | INDEX (10 bits) | BLOCK OFFSET (2 bits) ]
  offset  = log2(4 words) = 2 bits   (word-addressed offset inside block)
  index   = log2(1024)    = 10 bits
  tag     = 32 - 10 - 2 - 2  = 18 bits

States
  IDLE         – waiting for a CPU request
  COMPARE_TAG  – check valid/tag/dirty
  WRITE_BACK   – write dirty block to memory  (stalls MEMORY_LATENCY cycles)
  ALLOCATE     – read new block from memory   (stalls MEMORY_LATENCY cycles)
"""

import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BLOCK_WORDS    = 4          # words per block
WORD_BYTES     = 4          # bytes per word
BLOCK_BYTES    = BLOCK_WORDS * WORD_BYTES   # 16
NUM_BLOCKS     = 1024
WORD_OFFSET_BITS    = 2     # log2(4 words)
BYTE_OFFSET_BITS    = 2     # log2(4 bytes)
INDEX_BITS     = 10         # log2(1024)
TAG_BITS       = 18

MEMORY_LATENCY = 3          # cycles memory takes to complete a read or write


# ──────────────────────────────────────────────
# FSM States
# ──────────────────────────────────────────────
class State:
    IDLE        = "IDLE"
    COMPARE_TAG = "COMPARE_TAG"
    WRITE_BACK  = "WRITE_BACK"
    ALLOCATE    = "ALLOCATE"


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class CacheLine:
    valid: bool  = False
    dirty: bool  = False
    tag  : int   = 0
    data : list  = field(default_factory=lambda: [0] * BLOCK_WORDS)


@dataclass
class CPURequest:
    op   : str   # "READ" or "WRITE"
    addr : int   # 32-bit byte address
    data : Optional[int] = None   # only for WRITE


# ──────────────────────────────────────────────
# Memory model  (never misses, latency = N cycles)
# ──────────────────────────────────────────────
class Memory:
    def __init__(self, latency: int = MEMORY_LATENCY):
        self.latency      = latency
        self.store        : dict[int, list] = {}   # block_addr → [word0..word3]
        self.busy_cycles  = 0
        self.pending_op   = None   # ("READ"|"WRITE", block_addr, data?)

    def _block_addr(self, byte_addr: int) -> int:
        return (byte_addr >> (WORD_OFFSET_BITS + BYTE_OFFSET_BITS)) << (WORD_OFFSET_BITS + BYTE_OFFSET_BITS)
        # align to block boundary (WORD_BYTES * BLOCK_WORDS = 16 bytes)

    def start_read(self, byte_addr: int):
        ba = self._block_addr(byte_addr)
        self.pending_op  = ("READ", ba, None)
        self.busy_cycles = self.latency

    def start_write(self, byte_addr: int, block_data: list):
        ba = self._block_addr(byte_addr)
        self.pending_op  = ("WRITE", ba, list(block_data))
        self.busy_cycles = self.latency

    def tick(self) -> bool:
        """Advance one cycle.  Returns True when operation finishes."""
        if self.busy_cycles <= 0:
            return False
        self.busy_cycles -= 1
        if self.busy_cycles == 0:
            op, ba, data = self.pending_op
            if op == "WRITE":
                self.store[ba] = data
            self.pending_op = None
            return True
        return False

    @property
    def ready(self) -> bool:
        return self.busy_cycles == 0

    def read_block(self, byte_addr: int) -> list:
        ba = self._block_addr(byte_addr)
        return list(self.store.get(ba, [0] * BLOCK_WORDS))


# ──────────────────────────────────────────────
# Cache Controller FSM
# ──────────────────────────────────────────────
class CacheController:
    def __init__(self, memory: Memory):
        self.memory      = memory
        self.cache       = [CacheLine() for _ in range(NUM_BLOCKS)]
        self.state       = State.IDLE
        self.cycle       = 0

        # Current in-flight request
        self.req         : Optional[CPURequest] = None
        self.req_tag     = 0
        self.req_index   = 0
        self.req_offset  = 0   # word offset inside block
        self.initial_hit_miss = None  # Track initial HIT/MISS status

    # ── address decomposition ──
    def _decompose(self, addr: int):
        word_offset = (addr >> BYTE_OFFSET_BITS) & ((1 << WORD_OFFSET_BITS) - 1)   # bits [3:2]
        index       = (addr >> (WORD_OFFSET_BITS + BYTE_OFFSET_BITS)) & ((1 << INDEX_BITS) - 1)
        tag         = addr >> (WORD_OFFSET_BITS + BYTE_OFFSET_BITS + INDEX_BITS)
        return tag, index, word_offset

    # ── signal helpers ──
    def _signals(self) -> dict:
        line = self.cache[self.req_index] if self.req else None
        return {
            "state"        : self.state,
            "cpu_req"      : f"{self.req.op} @0x{self.req.addr:08X}" if self.req else "—",
            "tag_match"    : (line.tag == self.req_tag)   if (self.req and line) else "—",
            "valid"        : line.valid  if line else "—",
            "dirty"        : line.dirty  if line else "—",
            "mem_ready"    : self.memory.ready,
            "mem_busy_cyc" : self.memory.busy_cycles,
        }

    def _print_cycle_header(self):
        s = self._signals()
        print(f"\n{'─'*62}")
        print(f"  Cycle {self.cycle:>4}  │  State: {s['state']}")
        print(f"{'─'*62}")
        print(f"  CPU request  : {s['cpu_req']}")
        if self.req:
            print(f"  Tag / Index / Offset : {self.req_tag} / "
                  f"{self.req_index} / {self.req_offset}")
            print(f"  Cache line   : valid={s['valid']}  dirty={s['dirty']}  "
                  f"tag_match={s['tag_match']}")
        print(f"  Memory       : {'BUSY (' + str(s['mem_busy_cyc']) + ' cyc left)' if not s['mem_ready'] else 'READY'}")

    # ── one FSM tick ──
    def tick(self, new_req: Optional[CPURequest] = None) -> Optional[str]:
        """
        Drive the FSM one cycle.  Returns a result string ("HIT"/"MISS"/None)
        when the CPU request is fully satisfied.
        """
        self.cycle += 1
        mem_just_ready = self.memory.tick()   # advance memory first

        self._print_cycle_header()

        result = None

        # ── IDLE ──────────────────────────────────────────────────────────
        if self.state == State.IDLE:
            if new_req:
                self.req = new_req
                self.initial_hit_miss = None  # Reset for new request
                self.req_tag, self.req_index, self.req_offset = \
                    self._decompose(new_req.addr)
                print(f"  → CPU request accepted. Transitioning to COMPARE_TAG.")
                self.state = State.COMPARE_TAG
            else:
                print(f"  → No request. Staying IDLE.")

        # ── COMPARE_TAG ───────────────────────────────────────────────────
        elif self.state == State.COMPARE_TAG:
            line = self.cache[self.req_index]
            hit  = line.valid and (line.tag == self.req_tag)

            # Record initial hit/miss on first COMPARE_TAG evaluation
            if self.initial_hit_miss is None:
                self.initial_hit_miss = "HIT" if hit else "MISS"

            if hit:
                # ── HIT ──
                if self.req.op == "READ":
                    value = line.data[self.req_offset]
                    print(f"  → CACHE HIT (READ).  Returning word={value}  "
                          f"(no memory traffic).")
                    result = f"INITIAL: {self.initial_hit_miss}  |  READ  word={value}"
                else:  # WRITE
                    old = line.data[self.req_offset]
                    line.data[self.req_offset] = self.req.data
                    line.dirty = True
                    print(f"  → CACHE HIT (WRITE). Updated word {old}→"
                          f"{self.req.data}.  Block marked DIRTY.")
                    result = f"INITIAL: {self.initial_hit_miss}  |  WRITE word={self.req.data}"
                # Actions: Set Valid, Set Tag, if Write Set Dirty
                line.valid = True
                line.tag   = self.req_tag
                self.state = State.IDLE
                self.req   = None

            else:
                # ── MISS ──
                print(f"  → CACHE MISS ({'cold/clean' if not line.dirty else 'dirty'}).")
                if line.valid and line.dirty:
                    # reconstruct block byte address from index + old tag
                    old_block_addr = (line.tag << (INDEX_BITS + WORD_OFFSET_BITS + BYTE_OFFSET_BITS)) | \
                                     (self.req_index << (WORD_OFFSET_BITS + BYTE_OFFSET_BITS))
                    print(f"  → Old block is DIRTY.  Starting WRITE-BACK to "
                          f"@0x{old_block_addr:08X}.  Transitioning to WRITE_BACK.")
                    self.memory.start_write(old_block_addr, line.data)
                    self._wb_addr = old_block_addr
                    self.state = State.WRITE_BACK
                else:
                    print(f"  → Old block is CLEAN (or invalid).  "
                          f"Starting ALLOCATE.  Transitioning to ALLOCATE.")
                    self.memory.start_read(self.req.addr)
                    self.state = State.ALLOCATE

        # ── WRITE_BACK ────────────────────────────────────────────────────
        elif self.state == State.WRITE_BACK:
            if not mem_just_ready:
                print(f"  → Memory NOT ready.  Stalling in WRITE_BACK "
                      f"({self.memory.busy_cycles} cycles remaining).")
            else:
                print(f"  → Write-back complete.  Old block flushed to memory.")
                print(f"  → Starting ALLOCATE (read new block).  "
                      f"Transitioning to ALLOCATE.")
                self.memory.start_read(self.req.addr)
                self.state = State.ALLOCATE

        # ── ALLOCATE ──────────────────────────────────────────────────────
        elif self.state == State.ALLOCATE:
            if not mem_just_ready:
                print(f"  → Memory NOT ready.  Stalling in ALLOCATE "
                      f"({self.memory.busy_cycles} cycles remaining).")
            else:
                new_block = self.memory.read_block(self.req.addr)
                line = self.cache[self.req_index]
                line.data  = new_block
                line.valid = True
                line.dirty = False
                line.tag   = self.req_tag
                print(f"  → Allocation complete.  Block loaded: {new_block}.")
                print(f"  → Transitioning to COMPARE_TAG to re-evaluate.")
                self.state = State.COMPARE_TAG

        return result

    def run_request(self, req: CPURequest) -> str:
        """Run the FSM until the request is fully satisfied. Returns result string."""
        print(f"\n{'═'*62}")
        print(f"  NEW CPU REQUEST: {req.op} @ 0x{req.addr:08X}"
              + (f"  data={req.data}" if req.data is not None else ""))
        print(f"{'═'*62}")

        result = None
        first  = True
        while result is None:
            result = self.tick(new_req=req if first else None)
            first  = False
        print(f"\n  ✔  Request resolved in cycle {self.cycle}.  Result: {result}")
        return result


# ──────────────────────────────────────────────
# Demo scenario
# ──────────────────────────────────────────────
def main():
    print(textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════╗
    ║        Cache Controller FSM Simulation                   ║
    ║  Block=16 B | Cache=16 KB (1024 blocks) | Addr=32-bit    ║
    ║  Memory latency = 3 cycles                               ║
    ╚══════════════════════════════════════════════════════════╝
    """))

    mem   = Memory(latency=MEMORY_LATENCY)
    ctrl  = CacheController(mem)

    # Pre-populate memory with recognisable data so reads return non-zero values
    
    mem.store[0x00000000] = [0xA0, 0xA1, 0xA2, 0xA3] # Block at address 0x0000_0000  (index=0, tag=0)
    mem.store[0x00000010] = [0xB0, 0xB1, 0xB2, 0xB3] # Block at address 0x0000_0010  (index=1, tag=0)
    mem.store[0x00100000] = [0xC0, 0xC1, 0xC2, 0xC3] # Block at address 0x0010_0000  (index=0, tag=1)

    # ── Request sequence ──────────────────────────────────────────────────
    # Addr breakdown reminder:
    #   0x00000000 → tag=0,   index=0,  offset=0
    #   0x00000004 → tag=0,   index=0,  offset=1
    #   0x00000010 → tag=0,   index=1,  offset=0
    #   0x00100000 → tag=1,   index=0,  offset=0
    requests = [
        # 1. Cold miss – clean eviction, read block into index 0
        CPURequest("READ",  0x00000000),
        # 2. Hit – same block, different word
        CPURequest("READ",  0x00000004),
        # 3. Write hit – dirty the block at index 0
        CPURequest("WRITE", 0x00000004, data=0xFF),
        # 4. Miss – different index, cold miss into index 1
        CPURequest("READ",  0x00000010),
        # 5. Conflict miss – tag=1 vs dirty tag=0 at index 0 → write-back then allocate
        CPURequest("READ",  0x00100000),
        # 6. Write to freshly allocated block (now clean, index 0 tag 1)
        CPURequest("WRITE", 0x00100000, data=0x42),
        # 7. Read same word back – should hit and return 0x42
        CPURequest("READ",  0x00100000),
    ]

    summary = []
    for i, req in enumerate(requests, 1):
        print(f"\n{'▓'*62}")
        print(f"  Request #{i}")
        res = ctrl.run_request(req)
        summary.append((i, req.op, req.addr, res))

    # ── Summary table ──
    print(f"\n\n{'═'*62}")
    print(f"  SIMULATION SUMMARY")
    print(f"{'═'*62}")
    print(f"  {'#':>2}  {'Op':<6} {'Address':>12}  Result")
    print(f"  {'─'*2}  {'─'*6} {'─'*12}  {'─'*30}")
    for idx, op, addr, res in summary:
        print(f"  {idx:>2}  {op:<6} 0x{addr:08X}  {res}")
    print(f"{'═'*62}")
    print(f"  Total cycles elapsed: {ctrl.cycle}")
    print()


if __name__ == "__main__":
    main()