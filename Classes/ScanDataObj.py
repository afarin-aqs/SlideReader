from copy import deepcopy
import inspect

class ScanData:
    def __init__(
            self, file_name, clusters_dict=None, blocks_dict=None, backup_clusters_dict=None,
            backup_blocks_dict=None, preprocess_params=None, circle_finding_params_hough=None,
            clustering_params_DBSCAN=None, cAb_names=None, avg_spot_r=20, avg_spot_distance=25, 
            default_assay='SD4', default_scan_size=5, default_block_ncol=4, default_block_nrow=16,
            default_block_size=800, predicted_clusters_ids=None, sorted_circles=None
    ):
        self.file_name = file_name
        self.assay = default_assay
        self.scan_size = default_scan_size
        self.block_ncol = default_block_ncol
        self.block_nrow = default_block_nrow
        self.avg_spot_r = avg_spot_r
        self.avg_spot_distance = avg_spot_distance
        self.block_size = default_block_size
        if not clusters_dict:
            self.clusters_dict = {}
        if not blocks_dict:
            self.blocks_dict = {}
        if not backup_clusters_dict:
            self.backup_clusters_dict = {}
        if not backup_blocks_dict:
            self.backup_blocks_dict = {}
        if not cAb_names:
            self.cAb_names = []
        if not predicted_clusters_ids:
            self.predicted_clusters_ids = []
        if not sorted_circles:
            self.sorted_circles = []
        if not preprocess_params:
            self.preprocess_params = {
                'blur_kernel_size': 19,
                'contrast_thr': 650,
                'canny_edge_thr1': 100,
                'canny_edge_thr2': 10
            }
        if not circle_finding_params_hough:
            self.circle_finding_params_hough = {
                'method_name': 'Hough',
                'dp': 1.1,
                'minDist': 40,
                'param1': 19,
                'param2': 21,
                'minRadius': 14,
                'maxRadius': 22,
            }
        if not clustering_params_DBSCAN:
            self.clustering_params_DBSCAN = {
                'eps': 1200,  # lower means harder
                'min_samples': 4,
                'x_power': 3,
                'y_power': 7,
                # 'extra_y_cost': True
            }

    def get_clusters_dict(self):
        return self.clusters_dict

    def get_blocks_dict(self):
        return self.blocks_dict

    def get_backup_clusters_dict(self):
        return self.backup_clusters_dict

    def get_backup_blocks_dict(self):
        return self.backup_blocks_dict

    def reset_clusters_dict(self):
        self.clusters_dict = {}
        self.backup_clusters_dict = {}

    def reset_blocks_dict(self):
        self.blocks_dict = {}
        self.backup_blocks_dict = {}


    def set_assay_scan_size_dependent_params(self, debug=False):
        if self.assay == 'OF':
            self.block_ncol = 3
            self.block_nrow = 8
            if self.scan_size == 10:
                self.block_size = 500
            elif self.scan_size == 5:
                self.block_size = 920
        elif self.assay == 'SD4':
            self.block_ncol = 4
            self.block_nrow = 16
            if self.scan_size == 10:
                self.block_size = 400
            elif self.scan_size == 5:
                self.block_size = 800
        else:
            print(f'assay cannot be {self.assay}! -> either OF or SD4')
        if debug:
            print(self.__dict__)

    def set_new_params(self, new_params_dict,debug=False):
        for key, value in new_params_dict.items():
            setattr(self, key, value)
        if debug:
            print(self.__dict__)

    def add_new_cluster_to_dict(self, cluster):
        self.clusters_dict[cluster.cluster_id] = cluster

    def add_new_block_to_dict(self, block):
        self.blocks_dict[block.block_id] = block

    def get_cluster(self, cluster_id):
        if cluster_id in self.clusters_dict:
            return self.clusters_dict[cluster_id]
        caller_frame = inspect.currentframe().f_back
        line_no = caller_frame.f_lineno  # Line number where custom_print was called
        func_name = caller_frame.f_code.co_name  # Name of the function that called custom_print
        caller_caller_frame = caller_frame.f_back if caller_frame else None
        caller_caller_func_name = caller_caller_frame.f_code.co_name
        caller_caller_line_no = caller_caller_frame.f_lineno
        text = f"[{caller_caller_func_name[:20].ljust(20)}({caller_caller_line_no}) - {func_name[:20].ljust(20)}({line_no})]"
        # print(f'{text}: cluster {cluster_id} does not exist')
        if cluster_id in self.backup_clusters_dict:
            return self.backup_clusters_dict[cluster_id]
        print(f'{text}: cluster {cluster_id} does not exist, even in backup clusters dictionary!!')


    def get_block(self, block_id: str):
        return self.blocks_dict[block_id]

    def get_cluster_backup(self, cluster_id):
        return deepcopy(self.backup_clusters_dict[cluster_id])

    # def get_block_backup(self, block_id):
    #     return deepcopy(self.backup_blocks_dict[block_id])

    # def save_cluster_backup(self, cluster_id):
    #     self.backup_clusters_dict[cluster_id] = self.get_cluster(cluster_id)

    # def save_block_backup(self, block_id):
    #     self.backup_blocks_dict[block_id] = self.get_block(block_id)

    def save_clusters_dict_backup(self):
        self.backup_clusters_dict = deepcopy(self.clusters_dict)

    def save_blocks_dict_backup(self):
        self.backup_blocks_dict = deepcopy(self.blocks_dict)

    def delete_cluster(self, cluster_id):
        if cluster_id not in self.clusters_dict:
            # print(f'cluster{cluster_id} is already deleted!')
            return
        del self.clusters_dict[cluster_id]


all_scan_data = {}


def add_new_scan_data_to_dict(scan_data,debug=False):
    if scan_data.file_name in all_scan_data and debug:
        print(f"scan data for {scan_data.file_name} is already in the all_scan_data, so I'm overwriting it!")
    all_scan_data[scan_data.file_name] = scan_data
    return

def create_new_scan_data(file_name,debug=False):
    if debug:
        print(f'starting a new scan data for {file_name}')
    scan_data = ScanData(file_name=file_name)
    add_new_scan_data_to_dict(scan_data=scan_data,debug=debug)
    if debug:
        print(f'This is the new scan data: {scan_data.__dict__}')
    return scan_data

def get_scan_data(file_name:str) -> ScanData:
    if file_name not in all_scan_data:
        print(f'{file_name} not in all_scan_data. returning a new scan data!')
        return None
    return all_scan_data[file_name]

def update_scan_data_dict(scan_data):
    if scan_data.file_name not in all_scan_data:
        print(f'There are not scan data with file_name {scan_data.file_name} in the dict!!!')
        print(f'These are the files in scan_data dict: {all_scan_data.keys()}')
    all_scan_data[scan_data.file_name] = scan_data


def init_or_reset_params(reset=False, file_name=None, input_param_dict=None, debug=False):
    scan_data = get_scan_data(file_name)
    if scan_data is not None and not reset:
        print(f'Skipping param initiation because they are already loaded from pickle files for {file_name}')
        return

    if scan_data is None:
        scan_data = create_new_scan_data(file_name=file_name, debug=debug)

    if input_param_dict is None:
        input_param_dict = {} #anything not given as input, will be set as its default value

    key_name_correction = {
        'preprocess_params': ['pp','image_preprocessing_params'],
        'circle_finding_params_hough': ['cf','hough_circle_finding_params', 'circle_finding_params','hough_params'],
        'clustering_params_DBSCAN': ['cl','clustering_params', 'DBSCAN_clustering_params','DBSCAN_params'],
    }
    for right_key, list_of_possible_names in key_name_correction.items():
        for wrong_key in list_of_possible_names:
            if wrong_key in input_param_dict.keys():
                input_param_dict[right_key] = input_param_dict.pop(wrong_key)

    scan_data.set_new_params(input_param_dict,debug=debug)
    scan_data.set_assay_scan_size_dependent_params(debug=debug)
    update_scan_data_dict(scan_data)
    print(f'Successfully set the params for {file_name} :)')
    return


#%% Section two is images!
images_dict = {}

def add_to_images_dict(file_name, dict_key, dict_value):
    if file_name not in images_dict:
        images_dict[file_name] = {}
    images_dict[file_name][dict_key] = deepcopy(dict_value)

def add_block_related_images_to_dict(file_name, block_id, image_tag, image_file):
    if file_name not in images_dict:
        images_dict[file_name] = {}
    if block_id not in images_dict[file_name]:
        images_dict[file_name][block_id] = {}
    images_dict[file_name][block_id][image_tag] = deepcopy(image_file)

def get_image_from_dict(file_name, dict_key):
    if file_name not in images_dict:
        print(f'There are no images with file_name {file_name} in the dict!!!')
        return None
    if dict_key not in images_dict[file_name]:
        print(f'There is no image with dict_key {dict_key} in the dict[{file_name}]!!!')
        return None
    return deepcopy(images_dict[file_name][dict_key])


def get_block_image(file_name, block_id, image_tag=None):
    if file_name not in images_dict:
        print(f'file_name {file_name} is not in the dict!!!')
        return None
    if block_id not in images_dict[file_name]:
        print(f'{block_id} is not here... :| !!!')
        return None
    block_images = images_dict[file_name][block_id]
    if image_tag not in block_images:
        print(f'{image_tag} is not here... :| !!!')
        return None
    return block_images[image_tag]



