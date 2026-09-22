"""Educational DEFLATE/zlib codec implemented without compression libraries.

Encoder uses LZ77 and fixed Huffman codes (stored blocks at level zero).
Decoder supports stored, fixed and dynamic blocks from RFC 1951.
"""

from bisect import bisect_right
from collections import deque

LENGTH_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27,
               31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]
LENGTH_EXTRA = [0]*8 + [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4 + [0]
DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
             193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
             6145, 8193, 12289, 16385, 24577]
DIST_EXTRA = [0]*4 + [i for i in range(1, 14) for _ in range(2)]
FIXED_LENGTHS = [8]*144 + [9]*112 + [7]*24 + [8]*8


def adler32(data):
    a, b = 1, 0
    for byte in data:
        a = (a + byte) % 65521
        b = (b + a) % 65521
    return (b << 16) | a


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.bits = self.count = 0

    def write(self, value, count):
        self.bits |= value << self.count
        self.count += count
        while self.count >= 8:
            self.data.append(self.bits & 255)
            self.bits >>= 8
            self.count -= 8

    def finish(self):
        if self.count:
            self.data.append(self.bits & 255)
        return bytes(self.data)


class BitReader:
    def __init__(self, data):
        self.data, self.position = data, 0

    def read(self, count):
        if self.position + count > len(self.data) * 8:
            raise ValueError("Truncated DEFLATE stream.")
        value = 0
        for i in range(count):
            value |= ((self.data[self.position // 8] >> (self.position % 8)) & 1) << i
            self.position += 1
        return value

    def align(self):
        self.position = (self.position + 7) // 8 * 8


def _codes(lengths):
    counts = [0]*16
    for length in lengths:
        if not 0 <= length <= 15:
            raise ValueError("Invalid Huffman code length.")
        if length:
            counts[length] += 1
    remaining = 1
    for count in counts[1:]:
        remaining = remaining * 2 - count
        if remaining < 0:
            raise ValueError("Oversubscribed Huffman tree.")
    if remaining and sum(counts) and not (counts[1] == 1 and sum(counts) == 1):
        raise ValueError("Incomplete Huffman tree.")
    next_code, code = [0]*16, 0
    for bits in range(1, 16):
        code = (code + counts[bits - 1]) << 1
        next_code[bits] = code
    result = {}
    for symbol, length in enumerate(lengths):
        if length:
            # Reverse canonical codes for the least-significant-bit writer.
            reverse = int(f"{next_code[length]:0{length}b}"[::-1], 2)
            result[symbol] = (reverse, length)
            next_code[length] += 1
    return result


def _tree(lengths):
    return {(code, length): symbol for symbol, (code, length) in _codes(lengths).items()}


def _symbol(reader, tree):
    code = 0
    for length in range(1, 16):
        code |= reader.read(1) << (length - 1)
        if (code, length) in tree:
            return tree[code, length]
    raise ValueError("Invalid Huffman symbol.")


def _stored(data):
    out = bytearray()
    for start in range(0, max(1, len(data)), 65535):
        block = data[start:start + 65535]
        size = len(block)
        out.append(int(start + size == len(data)))
        out.extend(size.to_bytes(2, "little"))
        out.extend((size ^ 65535).to_bytes(2, "little"))
        out.extend(block)
    return bytes(out)


def zlib_encode(data, level=9):
    """Levels 1–9 increase bounded LZ77 search depth; all levels are lossless."""
    if not isinstance(level, int) or not 0 <= level <= 9:
        raise ValueError("Compression level must be 0–9.")
    stored = _stored(data)
    if level == 0:
        body = stored
    else:
        writer = BitWriter()
        writer.write(1, 1)  # final block
        writer.write(1, 2)  # fixed Huffman
        literals, distances = _codes(FIXED_LENGTHS), _codes([5]*32)
        chains, window = {}, deque()
        position = 0
        while position < len(data):
            while window and position - window[0][0] > 32768:
                _, key = window.popleft()
                chains[key].popleft()
                if not chains[key]:
                    del chains[key]
            key = data[position:position + 3]
            best, distance = 0, 0
            if len(key) == 3:
                for attempt, candidate in enumerate(reversed(chains.get(key, ()))):
                    if attempt >= level * 8:
                        break
                    length, limit = 3, min(258, len(data) - position)
                    while length < limit and data[candidate + length] == data[position + length]:
                        length += 1
                    if length > best:
                        best, distance = length, position - candidate
                    if best == limit:
                        break
            step = best if best >= 3 else 1
            if best >= 3:
                index = bisect_right(LENGTH_BASE, best) - 1
                writer.write(*literals[257 + index])
                writer.write(best - LENGTH_BASE[index], LENGTH_EXTRA[index])
                index = bisect_right(DIST_BASE, distance) - 1
                writer.write(*distances[index])
                writer.write(distance - DIST_BASE[index], DIST_EXTRA[index])
            else:
                writer.write(*literals[data[position]])
            for offset in range(step):
                at = position + offset
                key = data[at:at + 3]
                if len(key) == 3:
                    chains.setdefault(key, deque()).append(at)
                    window.append((at, key))
            position += step
        writer.write(*literals[256])
        compressed = writer.finish()
        body = compressed if len(compressed) < len(stored) else stored
    return b"\x78\x01" + body + adler32(data).to_bytes(4, "big")


def _dynamic_trees(reader):
    nlit, ndist, ncode = reader.read(5) + 257, reader.read(5) + 1, reader.read(4) + 4
    if nlit > 286:
        raise ValueError("Invalid literal count.")
    order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    lengths = [0]*19
    for index in order[:ncode]:
        lengths[index] = reader.read(3)
    tree = _tree(lengths)
    expanded = []
    while len(expanded) < nlit + ndist:
        symbol = _symbol(reader, tree)
        if symbol < 16:
            expanded.append(symbol)
        elif symbol == 16:
            if not expanded:
                raise ValueError("No previous code length to repeat.")
            expanded.extend([expanded[-1]] * (reader.read(2) + 3))
        else:
            expanded.extend([0] * (reader.read(3) + 3 if symbol == 17 else reader.read(7) + 11))
        if len(expanded) > nlit + ndist:
            raise ValueError("Too many Huffman code lengths.")
    if not expanded[256]:
        raise ValueError("Missing end-of-block symbol.")
    return _tree(expanded[:nlit]), _tree(expanded[nlit:])


def zlib_decode(data, max_output):
    """Decode with output bounds, header validation, and an Adler-32 check."""
    if len(data) < 6 or data[0] & 15 != 8 or data[0] >> 4 > 7 or int.from_bytes(data[:2], "big") % 31 or data[1] & 32:
        raise ValueError("Invalid or unsupported zlib header.")
    reader, output = BitReader(data[2:-4]), bytearray()
    final = False
    while not final:
        final, kind = reader.read(1), reader.read(2)
        if kind == 0:
            reader.align()
            size, inverse = reader.read(16), reader.read(16)
            if size ^ inverse != 65535 or len(output) + size > max_output:
                raise ValueError("Invalid stored block length.")
            for _ in range(size):
                output.append(reader.read(8))
            continue
        if kind == 1:
            literals, distances = _tree(FIXED_LENGTHS), _tree([5]*32)
        elif kind == 2:
            literals, distances = _dynamic_trees(reader)
        else:
            raise ValueError("Reserved DEFLATE block type.")
        while True:
            symbol = _symbol(reader, literals)
            if symbol == 256:
                break
            if symbol < 256:
                if len(output) >= max_output:
                    raise ValueError("Decoded data exceeds expected size.")
                output.append(symbol)
            elif symbol <= 285:
                index = symbol - 257
                length = LENGTH_BASE[index] + reader.read(LENGTH_EXTRA[index])
                index = _symbol(reader, distances)
                if index >= 30:
                    raise ValueError("Reserved distance symbol.")
                distance = DIST_BASE[index] + reader.read(DIST_EXTRA[index])
                if distance > len(output) or distance > (1 << ((data[0] >> 4) + 8)) or len(output) + length > max_output:
                    raise ValueError("Invalid match distance or output size.")
                for _ in range(length):
                    output.append(output[-distance])
            else:
                raise ValueError("Reserved length symbol.")
    if (reader.position + 7) // 8 != len(reader.data) or adler32(output) != int.from_bytes(data[-4:], "big"):
        raise ValueError("Trailing compressed data or incorrect Adler-32 checksum.")
    return bytes(output)
