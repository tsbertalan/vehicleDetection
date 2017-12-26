from sklearn.model_selection._search import (
    check_cv, is_classifier, _check_multimetric_scoring, indexable, clone, 
    Parallel, partial, product, delayed, _fit_and_score, _aggregate_score_dicts,
    np, defaultdict, MaskedArray, rankdata
)
def fit(self, X, y=None, groups=None, **fit_params):
    """Run fit with all sets of parameters.

    Parameters
    ----------

    X : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples] or [n_samples, n_output], optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    **fit_params : dict of string -> object
        Parameters passed to the ``fit`` method of the estimator
    """
    if self.fit_params is not None:
        warnings.warn('"fit_params" as a constructor argument was '
                      'deprecated in version 0.19 and will be removed '
                      'in version 0.21. Pass fit parameters to the '
                      '"fit" method instead.', DeprecationWarning)
        if fit_params:
            warnings.warn('Ignoring fit_params passed as a constructor '
                          'argument in favor of keyword arguments to '
                          'the "fit" method.', RuntimeWarning)
        else:
            fit_params = self.fit_params
    estimator = self.estimator
    cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

    scorers, self.multimetric_ = _check_multimetric_scoring(
        self.estimator, scoring=self.scoring)

    if self.multimetric_:
        if self.refit is not False and (
                not isinstance(self.refit, six.string_types) or
                # This will work for both dict / list (tuple)
                self.refit not in scorers):
            raise ValueError("For multi-metric scoring, the parameter "
                             "refit must be set to a scorer key "
                             "to refit an estimator with the best "
                             "parameter setting on the whole data and "
                             "make the best_* attributes "
                             "available for that metric. If this is not "
                             "needed, refit should be set to False "
                             "explicitly. %r was passed." % self.refit)
        else:
            refit_metric = self.refit
    else:
        refit_metric = 'score'

    X, y, groups = indexable(X, y, groups)
    n_splits = cv.get_n_splits(X, y, groups)
    # Regenerate parameter iterable for each fit
    candidate_params = list(self._get_param_iterator())
    n_candidates = len(candidate_params)
    if self.verbose > 0:
        print("Fitting {0} folds for each of {1} candidates, totalling"
              " {2} fits".format(n_splits, n_candidates,
                                 n_candidates * n_splits))

    base_estimator = clone(self.estimator)
    pre_dispatch = self.pre_dispatch

    out = Parallel(
        n_jobs=self.n_jobs, verbose=self.verbose,
        pre_dispatch=pre_dispatch,
        n_tasks=n_candidates*n_splits,
    )(delayed(_fit_and_score)(clone(base_estimator), X, y, scorers, train,
                              test, self.verbose, parameters,
                              fit_params=fit_params,
                              return_train_score=self.return_train_score,
                              return_n_test_samples=True,
                              return_times=True, return_parameters=False,
                              error_score=self.error_score)
      for parameters, (train, test) in product(candidate_params,
                                               cv.split(X, y, groups)))

    # if one choose to see train score, "out" will contain train score info
    if self.return_train_score:
        (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
         score_time) = zip(*out)
    else:
        (test_score_dicts, test_sample_counts, fit_time,
         score_time) = zip(*out)

    # test_score_dicts and train_score dicts are lists of dictionaries and
    # we make them into dict of lists
    test_scores = _aggregate_score_dicts(test_score_dicts)
    if self.return_train_score:
        train_scores = _aggregate_score_dicts(train_score_dicts)

    results = dict()

    def _store(key_name, array, weights=None, splits=False, rank=False):
        """A small helper to store the scores/times to the cv_results_"""
        # When iterated first by splits, then by parameters
        # We want `array` to have `n_candidates` rows and `n_splits` cols.
        array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                          n_splits)
        if splits:
            for split_i in range(n_splits):
                # Uses closure to alter the results
                results["split%d_%s"
                        % (split_i, key_name)] = array[:, split_i]

        array_means = np.average(array, axis=1, weights=weights)
        results['mean_%s' % key_name] = array_means
        # Weighted std is not directly available in numpy
        array_stds = np.sqrt(np.average((array -
                                         array_means[:, np.newaxis]) ** 2,
                                        axis=1, weights=weights))
        results['std_%s' % key_name] = array_stds

        if rank:
            results["rank_%s" % key_name] = np.asarray(
                rankdata(-array_means, method='min'), dtype=np.int32)

    _store('fit_time', fit_time)
    _store('score_time', score_time)
    # Use one MaskedArray and mask all the places where the param is not
    # applicable for that candidate. Use defaultdict as each candidate may
    # not contain all the params
    param_results = defaultdict(partial(MaskedArray,
                                        np.empty(n_candidates,),
                                        mask=True,
                                        dtype=object))
    for cand_i, params in enumerate(candidate_params):
        for name, value in params.items():
            # An all masked empty array gets created for the key
            # `"param_%s" % name` at the first occurence of `name`.
            # Setting the value at an index also unmasks that index
            param_results["param_%s" % name][cand_i] = value

    results.update(param_results)
    # Store a list of param dicts at the key 'params'
    results['params'] = candidate_params

    # NOTE test_sample counts (weights) remain the same for all candidates
    test_sample_counts = np.array(test_sample_counts[:n_splits],
                                  dtype=np.int)
    for scorer_name in scorers.keys():
        # Computed the (weighted) mean and std for test scores alone
        _store('test_%s' % scorer_name, test_scores[scorer_name],
               splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_%s' % scorer_name, train_scores[scorer_name],
                   splits=True)

    # For multi-metric evaluation, store the best_index_, best_params_ and
    # best_score_ iff refit is one of the scorer names
    # In single metric evaluation, refit_metric is "score"
    if self.refit or not self.multimetric_:
        self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
        self.best_params_ = candidate_params[self.best_index_]
        self.best_score_ = results["mean_test_%s" % refit_metric][
            self.best_index_]

    if self.refit:
        self.best_estimator_ = clone(base_estimator).set_params(
            **self.best_params_)
        if y is not None:
            self.best_estimator_.fit(X, y, **fit_params)
        else:
            self.best_estimator_.fit(X, **fit_params)

    # Store the only scorer not as a dict for single metric evaluation
    self.scorer_ = scorers if self.multimetric_ else scorers['score']

    self.cv_results_ = results
    self.n_splits_ = n_splits

    return self

import sklearn.model_selection._search
sklearn.model_selection._search.BaseSearchCV.fit = fit

from sklearn.externals.joblib.parallel import (
    get_active_backend, _basestring, memstr_to_bytes, DEFAULT_MP_CONTEXT, 
    threading, itertools, time, _verbosity_filter, short_format_time
)

def __init__(
    self, n_jobs=1, backend=None, verbose=0, timeout=None,
    pre_dispatch='2 * n_jobs', batch_size='auto',
    temp_folder=None, max_nbytes='1M', mmap_mode='r', n_tasks=None
    ):
    active_backend, default_n_jobs = get_active_backend()
    if backend is None and n_jobs == 1:
        # If we are under a parallel_backend context manager, look up
        # the default number of jobs and use that instead:
        n_jobs = default_n_jobs
    self.n_jobs = n_jobs
    self.n_tasks = n_tasks
    self.verbose = verbose
    self.timeout = timeout
    self.pre_dispatch = pre_dispatch

    if isinstance(max_nbytes, _basestring):
        max_nbytes = memstr_to_bytes(max_nbytes)

    self._backend_args = dict(
        max_nbytes=max_nbytes,
        mmap_mode=mmap_mode,
        temp_folder=temp_folder,
        verbose=max(0, self.verbose - 50),
    )
    if DEFAULT_MP_CONTEXT is not None:
        self._backend_args['context'] = DEFAULT_MP_CONTEXT

    if backend is None:
        backend = active_backend
    elif isinstance(backend, ParallelBackendBase):
        # Use provided backend as is
        pass
    elif hasattr(backend, 'Pool') and hasattr(backend, 'Lock'):
        # Make it possible to pass a custom multiprocessing context as
        # backend to change the start method to forkserver or spawn or
        # preload modules on the forkserver helper process.
        self._backend_args['context'] = backend
        backend = MultiprocessingBackend()
    else:
        try:
            backend_factory = BACKENDS[backend]
        except KeyError:
            raise ValueError("Invalid backend: %s, expected one of %r"
                             % (backend, sorted(BACKENDS.keys())))
        backend = backend_factory()

    if (batch_size == 'auto' or isinstance(batch_size, Integral) and
            batch_size > 0):
        self.batch_size = batch_size
    else:
        raise ValueError(
            "batch_size must be 'auto' or a positive integer, got: %r"
            % batch_size)

    self._backend = backend
    self._output = None
    self._jobs = list()
    self._managed_backend = False

    # This lock is used coordinate the main thread of this process with
    # the async callback thread of our the pool.
    self._lock = threading.Lock()

def print_progress(self):
    """Display the process of the parallel execution only a fraction
       of time, controlled by self.verbose.
    """

    if hasattr(self, 'pbar'):
        self.pbar.update(self.n_completed_tasks - self.pbar.n)

    if not self.verbose:
        return
        
    elapsed_time = time.time() - self._start_time

    # Original job iterator becomes None once it has been fully
    # consumed : at this point we know the total number of jobs and we are
    # able to display an estimation of the remaining time based on already
    # completed jobs. Otherwise, we simply display the number of completed
    # tasks.
    if self._original_iterator is not None:
        if _verbosity_filter(self.n_dispatched_batches, self.verbose):
            return
        self._print('Done %3i tasks      | elapsed: %s',
                    (self.n_completed_tasks,
                     short_format_time(elapsed_time), ))
    else:
        index = self.n_completed_tasks
        # We are finished dispatching
        total_tasks = self.n_dispatched_tasks
        # We always display the first loop
        if not index == 0:
            # Display depending on the number of remaining items
            # A message as soon as we finish dispatching, cursor is 0
            cursor = (total_tasks - index + 1 -
                      self._pre_dispatch_amount)
            frequency = (total_tasks // self.verbose) + 1
            is_last_item = (index + 1 == total_tasks)
            if (is_last_item or cursor % frequency):
                return
        remaining_time = (elapsed_time / index) * \
                         (self.n_dispatched_tasks - index * 1.0)
        # only display status if remaining time is greater or equal to 0
        self._print('Done %3i out of %3i | elapsed: %s remaining: %s',
                    (index,
                     total_tasks,
                     short_format_time(elapsed_time),
                     short_format_time(remaining_time),
                     ))

def __call__(self, iterable):
    if self._jobs:
        raise ValueError('This Parallel instance is already running')
    # A flag used to abort the dispatching of jobs in case an
    # exception is found
    self._aborting = False
    if not self._managed_backend:
        n_jobs = self._initialize_backend()
    else:
        n_jobs = self._effective_n_jobs()

    ntasks = self.n_tasks
    if ntasks is None:
        try:
            ntasks = len(iterable)
        except TypeError:
            pass

    # Track progress.
    if (not hasattr(self, 'noProgressBar')) or (not self.noProgressBar):
        import tqdm
        # Are we in a notebook?
        import __main__ as main
        if not hasattr(main, '__file__'):
            self.pbar = tqdm.tqdm_notebook(total=ntasks)
        else:
            self.pbar = tqdm.tqdm(total=ntasks)

    iterator = iter(iterable)
    pre_dispatch = self.pre_dispatch

    if pre_dispatch == 'all' or n_jobs == 1:
        # prevent further dispatch via multiprocessing callback thread
        self._original_iterator = None
        self._pre_dispatch_amount = 0
    else:
        self._original_iterator = iterator
        if hasattr(pre_dispatch, 'endswith'):
            pre_dispatch = eval(pre_dispatch)
        self._pre_dispatch_amount = pre_dispatch = int(pre_dispatch)

        # The main thread will consume the first pre_dispatch items and
        # the remaining items will later be lazily dispatched by async
        # callbacks upon task completions.
        iterator = itertools.islice(iterator, pre_dispatch)

    self._start_time = time.time()
    self.n_dispatched_batches = 0
    self.n_dispatched_tasks = 0
    self.n_completed_tasks = 0
    try:
        # Only set self._iterating to True if at least a batch
        # was dispatched. In particular this covers the edge
        # case of Parallel used with an exhausted iterator.
        while self.dispatch_one_batch(iterator):
            self._iterating = True
        else:
            self._iterating = False

        if pre_dispatch == "all" or n_jobs == 1:
            # The iterable was consumed all at once by the above for loop.
            # No need to wait for async callbacks to trigger to
            # consumption.
            self._iterating = False
        self.retrieve()
        # Make sure that we get a last message telling us we are done
        elapsed_time = time.time() - self._start_time
        self._print('Done %3i out of %3i | elapsed: %s finished',
                    (len(self._output), len(self._output),
                     short_format_time(elapsed_time)))
    finally:
        if not self._managed_backend:
            self._terminate_backend()
        self._jobs = list()
    output = self._output
    self._output = None
    return output

import sklearn.externals.joblib.parallel

sklearn.externals.joblib.parallel.Parallel.__init__ = __init__
sklearn.externals.joblib.parallel.Parallel.print_progress = print_progress
sklearn.externals.joblib.parallel.Parallel.__call__ = __call__
