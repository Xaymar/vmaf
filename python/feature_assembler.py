__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

from vmaf_feature_extractor import VmafFeatureExtractor

class FeatureAssembler(object):
    """
    Assembles features for a input list of Assets on a input list of
    FeatureExtractors. For each asset, it outputs a dictionary of feature
    scores.
    """

    # add subclasses of FeatureExtractor to the list below for consideration
    FEATURE_EXTRACTOR_SUBCLASSES = [VmafFeatureExtractor,]

    def __init__(self, feature_dict, assets, logger, log_file_dir,
                 fifo_mode, delete_workdir, result_store):
        """
        :param feature_dict: in the format of:
        {FeatureExtractor_type:'all',},
        or
        {FeatureExtractor_type:[atom_features,],}.
        For example, the below are valid feature dicts:
        {'VMAF_feature':'all', 'BRISQUE_feature':'all'},
        {'VMAF_feature':['vif', 'ansnr'], 'BRISQUE_feature':'all'}
        :param assets:
        :param logger:
        :param log_file_dir:
        :param fifo_mode:
        :param delete_workdir:
        :param result_store:
        :return:
        """
        self.feature_dict = feature_dict
        self.assets = assets
        self.logger = logger
        self.log_file_dir = log_file_dir
        self.fifo_mode = fifo_mode
        self.delete_workdir = delete_workdir
        self.result_store = result_store

        self.type2results_dict = {}

    @property
    def ordered_scores_key_list(self):
        """
        CAUTION: order matters! ALWAYS use this ordered list to construct
        feature vector for TrainTestModel from result.
        :return:
        """
        scores_key_list = []
        for fextractor_type in sorted(self.feature_dict.keys()):
            for atom_feature in sorted(self._get_atom_features(fextractor_type)):

                scores_key = self._get_scores_key(fextractor_type, atom_feature)

                scores_key_list.append(scores_key)
        return scores_key_list

    def run(self):

        # for each FeatureExtractor_type key in feature_dict, find the subclass
        # of FeatureExtractor, run, and put results in a dict
        for fextractor_type in self.feature_dict:
            fextractor = self._get_fextractor_instance(fextractor_type)
            fextractor.run()
            self.type2results_dict[fextractor_type] = fextractor.results

        # assemble an output dict with demanded atom features
        # atom_features_dict = self.fextractor_atom_features_dict
        result_dicts = [dict() for _ in self.assets]
        for fextractor_type in self.feature_dict:
            assert fextractor_type in self.type2results_dict
            for atom_feature in self._get_atom_features(fextractor_type):
                scores_key = self._get_scores_key(fextractor_type, atom_feature)
                for result_index, result in enumerate(self.type2results_dict[
                                                          fextractor_type]):
                    result_dicts[result_index][scores_key] = result[scores_key]

        self.result_dicts = result_dicts

    def remove_logs(self):
        for fextractor_type in self.feature_dict:
            fextractor = self._get_fextractor_instance(fextractor_type)
            fextractor.remove_logs()

    def remove_results(self):
        for fextractor_type in self.feature_dict:
            fextractor = self._get_fextractor_instance(fextractor_type)
            fextractor.remove_results()

    def _get_scores_key(self, fextractor_type, atom_feature):
        fextractor_subclass = self._find_fextractor_subclass(fextractor_type)
        scores_key = fextractor_subclass._get_scores_key(atom_feature)
        return scores_key

    def _get_atom_features(self, fextractor_type):
        if self.feature_dict[fextractor_type] == 'all':
            fextractor_class = self._find_fextractor_subclass(fextractor_type)
            atom_features = fextractor_class.ATOM_FEATURES
        else:
            atom_features = self.feature_dict[fextractor_type]
        return atom_features

    def _get_fextractor_instance(self, fextractor_type):
        fextractor_class = self._find_fextractor_subclass(fextractor_type)
        fextractor = fextractor_class(self.assets,
                                      self.logger,
                                      self.log_file_dir,
                                      self.fifo_mode,
                                      self.delete_workdir,
                                      self.result_store)
        return fextractor

    @classmethod
    def _find_fextractor_subclass(cls, fextractor_type):
        matched_fextractor_subclasses = []
        for fextractor_subclass in cls.FEATURE_EXTRACTOR_SUBCLASSES:
            if fextractor_subclass.TYPE == fextractor_type:
                matched_fextractor_subclasses.append(fextractor_subclass)

        assert len(matched_fextractor_subclasses) == 1

        return matched_fextractor_subclasses[0]