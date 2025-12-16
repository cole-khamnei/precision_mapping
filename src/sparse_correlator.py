import numpy as np
import scipy

from tqdm.auto import tqdm

from . import spc_utils

# ----------------------------------------------------------------------------# 
# -----------------           Bock Analysis Module           -----------------# 
# ----------------------------------------------------------------------------# 


class BlockAnalysis:
    """
    """
    def __init__(self, data, backend="torch", device=None, symmetric=True, skip_diagonal=True):
        self.size = data.shape[1]
        self.shape = (data.shape[1], data.shape[1])
        
        self.backend = spc_utils.get_backend(backend)
        self.device_info = spc_utils.get_device_info(backend, device)

        self.skip_diagonal = skip_diagonal
        self.symmetric = symmetric
        self.preprocessed = False

        self.create_empty = lambda k=1: [self.backend.ones(0, **self.device_info) for _ in range(k)]

    @classmethod
    def run(cls, data,
            mask=None,
            exclude_index=None,
            block_size=4_000,
            backend="torch",
            device=None,
            leave=True,
            symmetric=False,
            dtype="float32",
            **block_params):
        """

        """
        backend = spc_utils.get_backend(backend)

        aggregator = cls(data, backend=backend, device=device, symmetric=symmetric, **block_params)
        device_info = spc_utils.get_device_info(backend, device)

        data = spc_utils.to_backend(backend, data, dtype=dtype, **device_info)

        # TODO: Fully test
        data = aggregator.preprocess(data)

        if exclude_index is not None:
            exclude_index = spc_utils.to_backend(backend, exclude_index, dtype="float16", **device_info)

        if symmetric and mask is not None:
            if spc_utils.get_nnz_safe(mask != mask.T) == 0:
                print("Mask is not symmetric, cannot use symmetric acceleration.")
                symmetric = False

        n_blocks = int(np.ceil(data.shape[1] / block_size))

        total_blocks = n_blocks * (n_blocks + 1) // 2 if symmetric else n_blocks ** 2
        pbar = tqdm(total=total_blocks * (block_size / 1000) ** 2, desc=f'{cls.__name__} Block Analysis', leave=leave)

        for a_count in range(n_blocks):
            a_start = a_count * block_size
            a_n_items = block_size if a_count < n_blocks - 1 else data.shape[1] - a_start
            a_index = slice(a_start, a_start + a_n_items)
            a_block = data[:, a_index]

            b_count_iter = range(a_count + 1) if symmetric else range(n_blocks)

            for b_count in b_count_iter:
                b_start = b_count * block_size
                b_n_items = block_size if b_count < n_blocks - 1 else data.shape[1] - b_start
                b_index = slice(b_start, b_start + b_n_items)
                b_block = data[:, b_index]

                aggregator(a_block, b_block, a_index, b_index, mask=mask, exclude_index=exclude_index)
                pbar.update(a_n_items * b_n_items // 1000 ** 2)
    
        pbar.update(pbar.total - pbar.n)
        pbar.close()
        return aggregator.results()

    def get_mask_chunk_index(self, mask, a_index, b_index, select_index):
        """ """
        if mask is not None:
            # print(mask.shape)
            mask_chunk = mask[a_index, b_index]
            if spc_utils.get_nnz_safe(mask_chunk) > 0:
                mask_flat = ~spc_utils.sparse_to_array(mask_chunk.astype(bool)).ravel()
                mask_flat = spc_utils.to_backend(self.backend, mask_flat, dtype=bool, **self.device_info)
                select_index &= mask_flat

        return select_index

    def get_exclude_chunk_index(self, exclude_index, a_index, b_index, select_index):
        """ """
        if exclude_index is not None:
            a_exclude = exclude_index[a_index]
            b_exclude = exclude_index[b_index]

            exclude_chunk = a_exclude.reshape(-1, 1) @ b_exclude.reshape(1, -1)
            select_index &= (exclude_chunk == 0).flatten()

        return select_index

    def preprocess(self, data):
        """ Can be modified child classes"""
        self.preprocessed = False
        return data

    def core_func(self, A, B, a_index, b_index):
        """ """
        raise NotImplementedError

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        raise NotImplementedError

    def results(self):
        return None


# ----------------------------------------------------------------------------# 
# -                Specific Aggregator And Correlator Classes                -# 
# ----------------------------------------------------------------------------# 


class SparseAggregator(BlockAnalysis):
    """
    keeps running list of coordinates of top p percent (or top n) sparse connections:

    """
    def __init__(self, *args, sparsity_percent=0.1, **kwargs):
        super().__init__(*args, **kwargs)

        self.sparsity_frac = sparsity_percent / 100

        # TODO: determine if sparsity percent should be before or after diagonal skip
        n_items = self.size ** 2 - self.size if self.skip_diagonal else self.size ** 2
        if self.symmetric:
            self.top_n = int(np.ceil(n_items * self.sparsity_frac)) // 2
        else:
            self.top_n = int(np.ceil(n_items * self.sparsity_frac))

        self.create_empty = lambda k=1: [self.backend.ones(0, **self.device_info) for _ in range(k)]

        self.min_tv = -self.backend.inf
        self.cache_tv, self.cache_ri, self.cache_ci = [], [], []
        self.compare_tv, self.compare_ri, self.compare_ci = self.create_empty(3)

    def compare_cache(self):
        """ """
        backend = self.backend
        self.compare_tv = backend.hstack(self.cache_tv + [self.compare_tv])
        self.compare_ri = backend.hstack(self.cache_ri + [self.compare_ri])
        self.compare_ci = backend.hstack(self.cache_ci + [self.compare_ci])
        self.cache_tv, self.cache_ri, self.cache_ci = [], [], []

        if len(self.compare_tv) > self.top_n:
            self.compare_tv, update_ti = spc_utils.MPS_safe_topk(backend, self.device_info, self.compare_tv, self.top_n)

            self.compare_ri = self.compare_ri[update_ti]
            self.compare_ci = self.compare_ci[update_ti]
            self.min_tv = self.compare_tv[-1]

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        backend = self.backend

        threshold_chunk_tv_index = backend.ones(A.shape[1] * B.shape[1], dtype=bool, **self.device_info)
        if exclude_index is not None:
            a_exclude = exclude_index[a_index]
            b_exclude = exclude_index[b_index]

            exclude_chunk = a_exclude.reshape(-1, 1) @ b_exclude.reshape(1, -1)
            threshold_chunk_tv_index &= (exclude_chunk == 0).flatten()

        if mask is not None:
            mask_chunk = mask[a_index, b_index]
            if spc_utils.get_nnz_safe(mask_chunk) > 0:
                # takes mask as bool, inverts (1/true specificies discard)
                mask_flat = ~spc_utils.sparse_to_array(mask_chunk.astype(bool)).ravel()
                mask_flat = spc_utils.to_backend(backend, mask_flat, dtype=bool, **self.device_info)
                threshold_chunk_tv_index &= mask_flat

        if not threshold_chunk_tv_index.any():
            return

        M_chunk = self.core_func(A, B, a_index, b_index)

        # TODO: Fix symmetry:
        if self.symmetric and a_index == b_index:
            # triu_index = backend.triu_indices(*M_chunk.shape, offset=1)
            triu_index = spc_utils.backend_triu_indices(backend, M_chunk.shape, offset=1)
            flat_triu = triu_index[0] * M_chunk.shape[0] + triu_index[1]
            threshold_chunk_tv_index[flat_triu] = 0

        chunk_tv = M_chunk.flatten()
        threshold_chunk_tv_index &= chunk_tv > self.min_tv

        chunk_ti = spc_utils.MPS_safe_where(backend, self.device_info, threshold_chunk_tv_index)
        chunk_ri, chunk_ci = spc_utils.MPS_safe_unravel_index(backend, self.device_info, chunk_ti, M_chunk.shape)

        if len(chunk_ti) == 0:
            return

        chunk_tv = chunk_tv[chunk_ti]
        chunk_ri += a_index.start
        chunk_ci += b_index.start

        self.cache_tv.append(chunk_tv)
        self.cache_ri.append(chunk_ri)
        self.cache_ci.append(chunk_ci)

        if sum(len(chunk) for chunk in self.cache_tv) > self.top_n:
            self.compare_cache()

    def results(self):
        self.compare_cache()

        assert len(self.cache_tv) == 0

        tv, ri, ci = spc_utils.to_np((self.compare_tv, self.compare_ri, self.compare_ci))
        if self.symmetric:
            non_diag_index = ri != ci
            tv = np.hstack([tv, tv[non_diag_index]])

            # following is purposely done in one line
            ri, ci = np.hstack([ri, ci[non_diag_index]]), np.hstack([ci, ri[non_diag_index]])

        return scipy.sparse.csr_matrix((tv, (ri.astype(int), ci.astype(int))), shape=self.shape)


class Correlator:
    def __init__(self, *args, backend="numpy", skip_diagonal="None", **kwargs):
        self.corr_func = spc_utils.backend_corr
        self.backend = spc_utils.get_backend(backend)
        self.skip_diagonal = skip_diagonal

    def preprocess(self, data):
        """ norm data so correlation is just multiplication """
        data = spc_utils.backend_norm(self.backend, data)
        self.preprocessed = True
        return data

    def core_func(self, A, B, a_index=None, b_index=None, preprocess_overide=False):
        """ """
        if self.preprocessed and not preprocess_overide:
            M_chunk = A.T @ B
        else:
            M_chunk = self.corr_func(self.backend, A, B)
        
        if a_index == b_index and self.skip_diagonal:
            M_chunk[self.backend.eye(M_chunk.shape[0], dtype=bool)] = -self.backend.inf

        return M_chunk


class SparseCorrelator(Correlator, SparseAggregator):
    """ """
    def __init__(self, *args, **kwargs):
        Correlator.__init__(self, *args, **kwargs)
        SparseAggregator.__init__(self, *args, **kwargs)


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
