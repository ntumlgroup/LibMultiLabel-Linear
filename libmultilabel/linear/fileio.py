from __future__ import annotations
import math

import mmap
import pathlib
import scipy.sparse as sparse
import numpy as np


class Array:
    def __init__(self,
                 path: str | pathlib.Path,
                 shape: int | tuple,
                 dtype: type,
                 exist_ok: bool = False,
                 ):
        self.path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=exist_ok)
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.size = math.prod(self.shape)
        self.file = open(path, 'r+b')
        self.file.truncate(self.size)
        self.mmap = mmap.mmap(
            self.file.fileno(),
            self.size * dtype.__itemsize__,
        )
        self.view = np.frombuffer(
            self.mmap,
            dtype=dtype,
        )

    def close(self):
        self.mmap.close()
        self.file.close()

    def __getitem__(self, *args):
        return self.view.__getitem__(*args)

    def __setitem__(self, *args):
        return self.view.__setitem__(*args)


class SpillCSR:
    def __init__(self,
                 prefix: str | pathlib.Path,
                 rows: int,
                 chunk_size: int,
                 dtype: type
                 ):
        self.prefix = pathlib.Path(prefix)
        self.chunk_size = chunk_size
        self.dtype = dtype

        # shape[1] is shared with spills
        self.shape = Array(f'{self.prefix}.shape', 2, dtype=np.int32)
        self.shape[0] = rows
        self.shape[1] = 0

        self.data = Array(
            f'{self.prefix}.data',
            shape=(rows, chunk_size),
            dtype=dtype,
        )
        self.indices = Array(
            f'{self.prefix}.indices',
            shape=(rows, chunk_size),
            dtype=np.int32,
        )
        self.endptr = Array(f'{self.prefix}.endptr',
                            shape=rows, dtype=np.int32)

        self.spill = np.full(rows, None, dtype=object)

    def append(self, arr: sparse.csr_matrix):
        if arr.shape[0] != self.rows:
            raise ValueError('dimension mismatch')

        self._append_parts(arr.data, arr.indices, arr.indptr)
        self.shape[1] += arr.shape[1]

    def _append_parts(self,
                      data: np.ndarray,
                      indices: np.ndarray,
                      indptr: np.ndarray,
                      ):
        for i in range(self.rows):
            begin = indptr[i]
            end = indptr[i+1]
            nnz = end - begin
            j = self.endptr[i]
            rem = self.chunk_size - j
            if rem >= nnz:
                self.data[i, j:j+nnz] = data[begin:end]
                self.indices[i, j:j+nnz] = indices[begin:end]
                self.indices[i, j:j+nnz] += self.shape[1]
                self.endptr[i] += nnz

            elif self.spill[i] is None:
                mid = begin + rem
                self.data[i, j:j+rem] = data[begin:mid]
                self.indices[i, j:j+rem] = indices[begin:mid]
                self.indices[i, j:j+rem] += self.shape[1]
                self.endptr[i] += rem

                spill = SpillCSR(
                    f'{self.prefix}.{i}',
                    1,
                    self.chunk_size,
                    self.dtype,
                )
                spill.shape = self.shape
                spill._append_parts(
                    data[mid:end],
                    indices[mid:end],
                    indptr[i:i+1],
                )
                self.spill[i] = spill

            else:
                self.spill[i]._append_parts(
                    data[begin:end],
                    indices[begin:end],
                    indptr[i:i+1],
                )

    def nnz(self) -> int:
        base = self.endptr.sum()
        spill = sum(s.nnz() for s in self.spill if s is not None)
        return base + spill

    def to_csr(self, prefix: str | pathlib.Path):
        nnz = self.nnz()
        data = Array(f'{prefix}.data', shape=nnz, dtype=self.dtype)
        indices = Array(f'{prefix}.indices', shape=nnz, dtype=np.int32)
        indptr = Array(f'{prefix}.indptr', shape=self.rows, dtype=np.int32)
        shape = Array(f'{prefix}.shape', shape=2, dtype=np.int32)
        shape[:] = self.shape

        top = 0
        for i in range(self.rows):
            indptr[i] = top
            top = self._marshal_to(data, indices, top, i)
        indptr[-1] = nnz

        return sparse.csr_matrix((data, indices, indptr), shape=(shape[0], shape[1]))

    def _marshal_to(self, data: Array, indices: Array, top: int, row: int) -> int:
        j = self.endptr[row]
        data[top:top+j] = self.data[row, :j]
        indices[top:top+j] = self.indices[row, :j]
        top += j

        spill = self.spill[row]
        if spill is not None:
            top = spill._marshal_to(data, indices, top, 0)

        return top
