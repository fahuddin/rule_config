import socket
from typing import Any, List, Optional, Union 

class MiniRedis:
    
    def __init__(self, host: str = "127.0.0.1", port: int = 6379, timeout: float = 3.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        
    
    def connect(self):
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.settimeout(self.timeout)
        return s
        
    def _encode(self, parts:List[Union[str, bytes, int]]):
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            if isinstance(p, int):
                b = str(p).encode()
            elif isinstance(p, bytes):
                b = p
            else:
                b = p.encode()
            out.append(f"${len(b)}\r\n".encode() + b + b"\r\n")
        return b"".join(out)
    
    def read_line(self, s:socket.socket) -> bytes:
        buff = bytearray()
        while True:
            ch = s.recv(1)
            if not ch:
                raise ConnectionError("Redis connection closed")
            buff += ch
            if len(buff) >= 2 and buff[-2:] == b"\r\n":
                return bytes(buff[:-2])
            
    def readexact(self, s: socket.socket, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            chunk = s.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Redis connection closed")
            data += chunk
        return bytes(data)

 
    def parse(self, s: socket.socket):
            prefix = self.readexact(s, 1)

            if prefix == b"+":  # simple string
                return self.read_line(s).decode()

            if prefix == b"-":  # error
                err = self.read_line(s).decode()
                raise RuntimeError(f"Redis error: {err}")

            if prefix == b":":  # integer
                return int(self.read_line(s))

            if prefix == b"$":  # bulk string
                ln = int(self.read_line(s))
                if ln == -1:
                    return None
                data = self.readexact(s, ln)
                self.readexact(s, 2)  # consume \r\n
                return data

            if prefix == b"*":  # array
                n = int(self.read_line(s))
                if n == -1:
                    return None
                return [self.parse(s) for _ in range(n)]

            raise RuntimeError(f"Unknown RESP prefix: {prefix!r}")
    
    def cmd(self, *parts):
        payload = self._encode(list(parts))
        s = self.connect()
        try:
            s.sendall(payload)
            return self.parse(s)
        finally:
            s.close()
     # Convenience methods
    def get(self, key: str) -> Optional[bytes]:
        return self.cmd("GET", key)  # type: ignore

    def set(self, key: str, value: Union[str, bytes]) -> str:
        if isinstance(value, str):
            value = value.encode()
        return self.cmd("SET", key, value)  # type: ignore

    def setex(self, key: str, ttl_seconds: int, value: Union[str, bytes]) -> str:
        if isinstance(value, str):
            value = value.encode()
        return self.cmd("SETEX", key, ttl_seconds, value)  # type: ignore

    def hset(self, key: str, field: str, value: Union[str, bytes]) -> int:
        if isinstance(value, str):
            value = value.encode()
        return int(self.cmd("HSET", key, field, value))  # type: ignore

    def hgetall(self, key: str) -> dict:
        arr = self.cmd("HGETALL", key)
        if arr is None:
            return {}
        items = list(arr)  # type: ignore
        out = {}
        for i in range(0, len(items), 2):
            k = items[i].decode()
            v = items[i + 1]
            out[k] = v.decode() if isinstance(v, (bytes, bytearray)) else v
        return out