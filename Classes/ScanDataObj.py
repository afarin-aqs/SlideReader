from copy import deepcopy
import inspect

class ScanData:
    def __init__(
            self, file_name, clusters_dict=None, blocks_dict=None, backup_clusters_dict=None,
            backup_blocks_dict=None,assay='SD4', scan_size=5, block_ncol=4, block_nrow=16, avg_spot_r=25,
            avg_spot_distance=25, block_size=700, cAb_names=None, preprocess_params=None, hough_params=None,
            DBSCAN_params=None
    ):
        self.file_name = file_name
        self.assay = assay
        self.scan_size = scan_size
        self.block_ncol = block_ncol
        self.block_nrow = block_nrow
        self.avg_spot_r = avg_spot_r
        self.avg_spot_distance = avg_spot_distance
        self.block_size = block_size
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
        if not preprocess_params:
            self.preprocess_params = {}
        if not hough_params:
            self.hough_params = {}
        if not DBSCAN_params:
            self.DBSCAN_params = {}

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


    def set_rest_of_params(self,debug=False):
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

        if self.scan_size == 10:
            self.avg_spot_r = 20  # radius of each spot (in pixels)
            self.avg_spot_distance = 20
        elif self.scan_size == 5:
            self.avg_spot_r= 19  # radius of each spot (in pixels)
            self.avg_spot_distance = 25
        else:
            print(f"don't have params for scan size {self.scan_size}")

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
        return None
    return all_scan_data[file_name]

def update_scan_data_dict(scan_data):
    if scan_data.file_name not in all_scan_data:
        print(f'There are not scan data with file_name {scan_data.file_name} in the dict!!!')
        print(f'These are the files in scan_data dict: {all_scan_data.keys()}')
    all_scan_data[scan_data.file_name] = scan_data


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



