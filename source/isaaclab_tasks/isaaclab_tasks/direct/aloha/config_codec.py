# config_codec.py
from typing import Dict, List, Tuple, Iterable
import hashlib

def _q3(p: Tuple[float,float,float], q: int=10) -> Tuple[int,int,int]:
    # квантование (0.1 м при q=10). Можешь поставить q=ratio для «в сетку»
    return (int(round(p[0]*q)), int(round(p[1]*q)), int(round(p[2]*q)))

class ConfigCodec:
    def __init__(self,
                 type_grids: Dict[str, List[Tuple[float,float,float]]],
                 types: Iterable[str] = ("movable_obstacle","static_obstacle"),
                 quant: int = 10,
                 max_radius: float | None = 7.0):
        self.types = tuple(types)
        self.quant = quant
        self.max_r2 = None if max_radius is None else max_radius*max_radius

        # КАТАЛОГ: для каждого типа — упорядоченный список квантизованных клеток
        self.catalog: Dict[str, List[Tuple[int,int,int]]] = {}
        self.idx_map: Dict[str, Dict[Tuple[int,int,int], int]] = {}

        for t in self.types:
            pts = type_grids.get(t, [])
            cats: List[Tuple[int,int,int]] = []
            seen = set()
            for x,y,z in pts:
                if self.max_r2 is not None and (x*x + y*y) > self.max_r2:
                    continue
                qp = _q3((x,y,z), self.quant)
                if qp in seen:  # защита от дублей
                    continue
                seen.add(qp)
                cats.append(qp)
            cats.sort()
            self.catalog[t] = cats
            self.idx_map[t] = {qp:i for i,qp in enumerate(cats)}

    def encode_bits(self, cfg: Dict[str, List[Tuple[float,float,float]]]) -> Dict[str, int]:
        """cfg: {type: [(x,y,z), ...]} → {type: bits_int}"""
        out: Dict[str,int] = {}
        for t in self.types:
            bits = 0
            imap = self.idx_map[t]
            for p in cfg.get(t, []):
                qp = _q3(p, self.quant)
                idx = imap.get(qp)
                if idx is not None:
                    bits |= (1 << idx)
            out[t] = bits
        return out

    def decode_bits(self, bits_by_type: Dict[str, int]) -> Dict[str, List[Tuple[float,float,float]]]:
        """Обратно в реальные координаты (деквантуем)."""
        out: Dict[str, List[Tuple[float,float,float]]] = {}
        inv_quant = 1.0 / self.quant
        for t in self.types:
            bits = bits_by_type.get(t, 0)
            cells = self.catalog[t]
            pts: List[Tuple[float,float,float]] = []
            b = bits
            while b:
                lsb = b & -b
                i = (lsb.bit_length() - 1)
                qx, qy, qz = cells[i]
                pts.append((qx*inv_quant, qy*inv_quant, qz*inv_quant))
                b ^= lsb
            out[t] = pts
        return out

    def encode_id_hex(self, cfg: Dict[str, List[Tuple[float,float,float]]]) -> str:
        """Детерминированный строковый ID типа 'm:<hex>|s:<hex>' в порядке self.types."""
        bits = self.encode_bits(cfg)
        parts = [f"{t[0]}:{bits[t]:x}" for t in self.types]  # первая буква типа + hex битсета
        return "|".join(parts)

    def encode_hash64(self, cfg: Dict[str, List[Tuple[float,float,float]]]) -> int:
        """Нехрупкий индекс (необратимый, но компактный): uint64 blake2b."""
        bits = self.encode_bits(cfg)
        payload = b"|".join(f"{t}:{bits[t]:x}".encode("ascii") for t in self.types)
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    def decode_id_hex(self, id_hex: str) -> Dict[str, List[Tuple[float,float,float]]]:
        """Обратимо, если ID формировался encode_id_hex и каталоги совпадают."""
        bits_by_type: Dict[str,int] = {}
        for chunk in id_hex.split("|"):
            if not chunk: continue
            pref, hexbits = chunk.split(":", 1)
            # сопоставляем первой букве правильный тип
            t = next(t for t in self.types if t.startswith(pref))
            bits_by_type[t] = int(hexbits, 16) if hexbits else 0
        return self.decode_bits(bits_by_type)
