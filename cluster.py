# THIS PROGRAM IS TO BE USED FOR EDUCATIONAL PURPOSES ONLY.
# CAN BE USED FOR INTERNAL PEN-TESTING, STAFF RECRUITMENT, SOCIAL ENGAGEMENT

import sklearn.pipeline
import sklearn.metrics
import sklearn.cluster
import datetime
import sklearn.metrics
import sklearn.grid_search
import sklearn.base
import sklearn.feature_extraction


def create_transformers():
    return [
        ('created_at', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'created_at')),
            ('preprocessor', CreatedAtPreprocessor()),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('followers_count', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_intfield, 'followers_count')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('listed_count', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_intfield, 'listed_count')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('favourites_count', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_intfield, 'favourites_count')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('statuses_count', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_intfield, 'statuses_count')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('friends_count', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_intfield, 'friends_count')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('location', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_location, 'location')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('profile_background_color', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'profile_background_color')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('profile_link_color', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'profile_link_color')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('profile_sidebar_fill_color', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field,
                                      'profile_sidebar_fill_color')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('profile_sidebar_border_color', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field,
                                      'profile_sidebar_border_color')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('profile_text_color', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'profile_text_color')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('verified', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'verified')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('default_profile_image', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'default_profile_image')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('default_profile', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'default_profile')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('geo_enabled', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'geo_enabled')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('contributors_enabled', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'contributors_enabled')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('protected', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'protected')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('is_translator', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'is_translator')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('lang', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'lang')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('time_zone', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_time_zone, 'time_zone')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('has_extended_profile', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'has_extended_profile')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('profile_use_background_image', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field,
                                      'profile_use_background_image')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('is_translation_enabled', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'is_translation_enabled')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ])),
        ('profile_background_tile', sklearn.pipeline.Pipeline([
            ('selector', ItemSelector(get_field, 'profile_background_tile')),
            ('vectorizer', sklearn.feature_extraction.DictVectorizer())
        ]))
    ]


class ItemSelector(sklearn.base.BaseEstimator,
                   sklearn.base.TransformerMixin):
    ''' For data grouped by feature, select subset of data '''
    def __init__(self, func, field_name=None):
        self.func = func
        self.field_name = field_name

    def fit(self, X, y=None):
        return self

    def transform(self, data_dict):
        return self.func(data_dict, self.field_name)


class CreatedAtPreprocessor(sklearn.base.BaseEstimator,
                            sklearn.base.TransformerMixin):
    ''' Preprocess features from created_at document '''
    def fit(self, X):
        return self

    def transform(self, corpus):
        for document in corpus:
            yield self._transform_document(document)

    def _transform_document(self, document,
                            hours_in_day=24, seconds_in_hour=3600):
        current_time = datetime.datetime.now()
        t_delta = current_time - self._convert(document['created_at'])
        document['created_at'] = \
            t_delta.days * hours_in_day * seconds_in_hour + t_delta.seconds
        return document

    def _convert(self, time_string):
        return datetime.datetime.strptime(time_string,
                                          "%a %b %d %H:%M:%S +0000 %Y")


def get_intfield(corpus, field_name):
    for document in corpus:
        yield {field_name: int(document[field_name])}


def get_field(corpus, field_name):
    for document in corpus:
        yield {field_name: document[field_name]}


def get_location(corpus, field_name):
    for document in corpus:
        if document[field_name]:
            yield {field_name: 1}
        else:
            yield {field_name: 0}


def get_time_zone(corpus, field_name):
    for document in corpus:
        if document[field_name]:
            yield {field_name: document[field_name]}
        else:
            yield {field_name: 'None'}


class Parameterize(sklearn.base.ClusterMixin):
    def __init__(self, scoring=sklearn.metrics.silhouette_score, n_iter=4):
        self.parameters = {
          'scoring': scoring,
          'n_iter': n_iter
        }

    def clusterer_choices(self):
        parameter_distributions = {
            sklearn.cluster.KMeans: {
                'n_clusters': [2, 3, 4, 5],
                'init': ['k-means++'],
                'n_init': [10],
                'max_iter': [300],
                'tol': [0.0001],
                'precompute_distances': ['auto']
            },
            sklearn.cluster.Birch: {
                'threshold': [0.1, 0.50],
                'branching_factor': [50],
                'n_clusters': [2, 3, 4, 5],
                'compute_labels': [True],
                'copy': [True]
            }
        }
        return parameter_distributions.items()

    def fit(self, X, y=None):
        silhouette_scores = {}
        scored_models = {}
        for clusterer_algo, clusterer_hyperparams in self.clusterer_choices():
            for hyperparam_grid in list(sklearn.grid_search.ParameterSampler(
                    clusterer_hyperparams, n_iter=self.parameters['n_iter'])):
                clusterer = clusterer_algo(**hyperparam_grid)
                cluster_labels = clusterer.fit_predict(X)
                silhouette_scores[clusterer] = \
                    self.parameters['scoring'](X, cluster_labels)
                scored_models[clusterer] = clusterer
        self.cluster_choice = scored_models[max(silhouette_scores,
                                                key=silhouette_scores.get)]
        return self

    def predict(self, X, y=None):
        return self.cluster_choice.predict(X)
