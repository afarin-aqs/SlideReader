from importlib import reload
import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import cv2
from copy import deepcopy
import traceback
from Functions import ClassesFunctions, CommonFunctions
from Classes import ScanDataObj
reload(ClassesFunctions)
reload(CommonFunctions)
reload(ScanDataObj)
from Functions.CommonFunctions import debug_report


class Block:
    def __init__(self, block_id, file_name, row_number=None, col_number=None, start_x=None, end_x=None, start_y=None,
                 end_y=None, Ag_conc=None, clusters_ids_list=None, backup_clusters_ids_list=None,
                 mask_start_coords=None, cAb_names=None, dAb_map_key=None, dAb_label=None,
                 sorted_clusters_ids_list=None, target=None, results_counts=None, fg_bg_calculated_flag=False,
                 intensities_dict_list=None, backup_start_x=None, backup_start_y=None, min_max_coords_of_clusters=None,
                 mask_end_coords=None, dont_touch_this_block=None):

        self.block_id = block_id
        self.row_number = row_number
        self.col_number = col_number
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.Ag_conc = Ag_conc
        self.file_name = file_name
        self.cAb_names = cAb_names
        self.dAb_map_key = dAb_map_key
        self.dAb_label = dAb_label
        self.target = target
        self.results_counts = results_counts
        self.fg_bg_calculated_flag = fg_bg_calculated_flag
        self.backup_start_x = backup_start_x
        self.backup_start_y = backup_start_y
        self.dont_touch_this_block = dont_touch_this_block

        if clusters_ids_list is None:
            clusters_ids_list = []
        if backup_clusters_ids_list is None:
            backup_clusters_ids_list = []
        if sorted_clusters_ids_list is None:
            sorted_clusters_ids_list = []
        if intensities_dict_list is None:
            intensities_dict_list = []
        if mask_start_coords is None:
            mask_start_coords = [0, 0]
        if mask_end_coords is None:
            mask_end_coords = [0, 0]
        if min_max_coords_of_clusters is None:
            min_max_coords_of_clusters = {'min_x': None, 'max_x': None, 'min_y': None, 'max_y': None}

        self.clusters_ids_list = clusters_ids_list
        self.backup_clusters_ids_list = backup_clusters_ids_list
        self.sorted_clusters_ids_list = sorted_clusters_ids_list
        self.intensities_dict_list = intensities_dict_list
        self.mask_start_coords = mask_start_coords  # this is relative to block start coords
        self.mask_end_coords = mask_end_coords  # this is relative to block start coords
        self.min_max_coords_of_clusters = min_max_coords_of_clusters


    def set_start_and_end_of_block(self, init_offset=None, block_size_adjustment=None,
                                   block_distance_adjustment=None, debug=False):
        # this function sets start and end coords of a block, for the first time.
        # it would only look at ncol, nrow, and scan size.
        # no backup or image is saved here.
        debug_report(f'** running set_start_and_end_of_block for block{self.block_id}', debug)

        if not init_offset:
            init_offset = [0,0]
        if not block_size_adjustment:
            block_size_adjustment = [0,0]
        if not block_distance_adjustment:
            block_distance_adjustment = [0,0]
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        scan_size = data_obj.scan_size
        block_size = data_obj.block_size
        block_ncol = data_obj.block_ncol
        debug_report(f'scan_size={scan_size}, block_size={block_size}', debug)
        debug_report(f'in the beginning x: {(self.start_x, self.end_x)}, y: {(self.start_y, self.end_y)}', debug)

        if scan_size == 10 and block_ncol == 3:
            distance = [150,100]
            offset = [10,100]

        elif scan_size == 5 and block_ncol == 3:
            distance = [660,890]
            offset = [100,400]

        elif scan_size == 5 and block_ncol == 4:
            distance = [90,45]
            offset = [300, 40]
        else:
            distance = [0,0]
            offset = [0,0]

        block_distance = [x + y for x, y in zip(distance, block_distance_adjustment)]
        first_block_offset = [x + y for x, y in zip(offset, init_offset)]

        self.start_x = int(self.col_number*(block_size+block_distance[0]) + first_block_offset[0])
        self.start_y = int(self.row_number*(block_size+block_distance[1]) + first_block_offset[1])

        self.end_x = self.start_x + int(block_size + block_size_adjustment[0])
        self.end_y = self.start_y + int(block_size + block_size_adjustment[1])
        debug_report(f'in the end x: {(self.start_x, self.end_x)}, y: {(self.start_y, self.end_y)}', debug)
        return

    def add_cropped_images(self, debug=False, plot_images=False):
        # this function is called when the block has it's start and end coords.

        # debug=True
        debug_report(f'** running add_cropped_images for block{self.block_id}', debug)

        data_obj = ScanDataObj.get_scan_data(self.file_name)
        block_size = data_obj.block_size
        debug_report(f'block size={block_size}', debug)

        file_image = ScanDataObj.get_image_from_dict(file_name=self.file_name, dict_key='file_image')
        file_scaled_image = ScanDataObj.get_image_from_dict(file_name=self.file_name, dict_key='file_scaled_image')
        # neg_fg_mask = ScanDataObj.get_image_from_dict(file_name=self.file_name, dict_key='file_neg_fg_mask')

        if self.start_x is None or self.start_y is None or self.end_x is None or self.end_y is None:
            debug_report(f'at least one of the coords is none! x: {(self.start_x, self.end_x)} and y :{(self.start_x, self.end_x)}', debug)
            self.set_start_and_end_of_block(debug)

        block_image = file_image[self.start_y:self.end_y,
                      self.start_x:self.end_x] if file_image.any() else None
        block_scaled_image = file_scaled_image[self.start_y:self.end_y,
                             self.start_x:self.end_x] if file_scaled_image.any() else None
        # block_neg_fg_mask = neg_fg_mask[self.start_y:self.end_y,
        #                     self.start_x:self.end_x] if file_scaled_image.any() else None

        ScanDataObj.add_block_related_images_to_dict(
            file_name=self.file_name,
            block_id=self.block_id,
            image_tag='image',
            image_file=block_image
        )
        ScanDataObj.add_block_related_images_to_dict(
            file_name=self.file_name,
            block_id=self.block_id,
            image_tag='scaled_image',
            image_file=block_scaled_image
        )

        # 'neg_fg_mask': block_neg_fg_mask,
        debug_report(f'added image{block_image.shape} and block_scaled_image{block_scaled_image.shape} to images_dict',debug)
        if plot_images:
            print(f'This is the block{self.block_id} image:')
            CommonFunctions.display_in_console(block_scaled_image)

    # B4
    def update_block_start_end_from_clusters_min_max(self, debug=False):
        # this function updates start and end coords of a block, based on clusters min_max (absolute) coords
        # then updates in_block_coords for all the clusters.
        # but will not add any images, and will not save backups.

        debug_report(f'** running update_block_start_end_from_clusters_min_max for block{self.block_id}',debug)
        debug_report(f'In the beginning x: {self.start_x}-{self.end_x} & y: {self.start_y}-{self.end_y}', debug)
        debug_report(f'Also: self.min_max_coords_of_clusters={self.min_max_coords_of_clusters} ', debug)

        for k, v in self.min_max_coords_of_clusters.items():
            if v is None:
                return
        # scan_size = ScanDataObj.get_scan_data(self.file_name).scan_size
        block_size = ScanDataObj.get_scan_data(self.file_name).block_size
        extra_padding = block_size//7 #checkme: this works for SD4 (size=5) -- need to check others...
        debug_report(f'extra_padding={extra_padding}', debug)

        #checkme
        min_max_copy = deepcopy(self.min_max_coords_of_clusters)
        start_x = int(max(min_max_copy['min_x'] - extra_padding, 0))
        start_y = int(max(min_max_copy['min_y'] - extra_padding, 0))
        end_x = int(max(min_max_copy['max_x'], (start_x + block_size)) + extra_padding)
        end_y = int(max(min_max_copy['max_y'], (start_y + block_size)) + extra_padding)
        self.reset_block_start_end_coords(start_coords=(start_x, start_y), end_coords=(end_x, end_y), debug=debug)
        debug_report(f'In the end x: {self.start_x}-{self.end_x} & y: {self.start_y}-{self.end_y}', debug)

    # B5
    def reset_block_start_end_coords(self, start_coords, end_coords, debug=False):
        ### this function will reset the start and end coords.
        # also makes adjustments for clusters
        # it will not add cropped image, and it will not save backups.

        #         debug=True
        debug_report(f'** B5: running reset_block_xy_definition for block{self.block_id}:\n{self.full_report(debug)}', debug)
        debug_report(f'In the beginning x: {self.start_x}-{self.end_x} & y: {self.start_y}-{self.end_y}', debug)
        debug_report(f'But we wanna reset them to -> start_coords={start_coords}, end_coords={end_coords}',debug)

        if start_coords == (self.start_x, self.start_y) and end_coords == (self.end_x, self.end_y):
            debug_report(f'NO NEED FOR UPDATE!', debug)
            return

        self.start_x, self.start_y = deepcopy(start_coords)
        self.end_x, self.end_y = deepcopy(end_coords)

        data_obj = ScanDataObj.get_scan_data(self.file_name)
        for cluster_id in self.clusters_ids_list:
            cluster = data_obj.get_cluster(cluster_id)
            if debug:
                cluster.full_report(return_str=debug)
            debug_report(f'Cluster{cluster_id} -> Before: spots_coords_in_block_list={cluster.spots_coords_in_block_list}', debug)
            cluster.add_block_info(block_id=self.block_id,debug=debug)
            cluster.add_spots_coords_in_block(debug=debug)
            debug_report(f'Cluster{cluster_id} -> After: spots_coords_in_block_list={cluster.spots_coords_in_block_list}', debug)
        # self.add_cropped_images(debug=debug,plot_images=debug)
        debug_report(f'{self.full_report(debug)}', debug)
        self.update_min_max_coords_of_clusters()
        self.add_cropped_images()

    def update_block_x_y_backup(self, debug=False):
        debug_report(f'** running update_block_x_y_backup for block{self.block_id}', debug)
        debug_report(f'BEFORE: self.backup_start={(self.backup_start_x, self.backup_start_y)}', debug)
        self.backup_start_x = deepcopy(self.start_x)
        self.backup_start_y = deepcopy(self.start_y)
        debug_report(f'AFTER: self.backup_start={(self.backup_start_x, self.backup_start_y)}', debug)

    def update_backup_clusters_ids_list(self, debug=False):
        debug_report(f'** running update_backup_clusters_ids_list for block{self.block_id}', debug)
        debug_report(f'BEFORE: self.backup_clusters_ids_list={self.backup_clusters_ids_list}', debug)
        self.backup_clusters_ids_list = deepcopy(self.clusters_ids_list)
        debug_report(f'AFTER: self.backup_clusters_ids_list={self.backup_clusters_ids_list}', debug)

    def save_backup(self, debug=False):
        self.update_block_x_y_backup(debug)
        self.update_backup_clusters_ids_list(debug)


    def redo_circle_finding(self, target_list, debug=False):
        debug_report(f'** running redo_circle_finding for block{self.block_id} with these targets: {target_list}', debug)

        block_image = ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='image')
        data_obj = ScanDataObj.get_scan_data(file_name=self.file_name)

        if self.block_id in target_list:
            debug_report(f'Redoing it for the whole block', debug)
            target_list = list(set(target_list + self.clusters_ids_list))
            debug_report(f'new target_list={target_list}', debug)

        for cluster_id in self.clusters_ids_list:
            if cluster_id not in target_list:
                continue
            cluster = data_obj.get_cluster(cluster_id)
            new_coords_list_but_in_block, highest_intensities_per_spot = CommonFunctions.optimized_spots_coords(
                input_image = block_image,
                coords_list = cluster.spots_coords_in_block_list,
                avg_r = data_obj.avg_spot_r,
                debug = debug
            )
            new_abs_coords_list = [new_coords_list_but_in_block[0][0] + self.start_x,
                                   new_coords_list_but_in_block[0][1] + self.start_y,
                                   new_coords_list_but_in_block[0][2]]
            cluster.add_new_spot_to_cluster(spot_coords=new_abs_coords_list, debug=debug)  # no backup is saved
            self.add_cluster_related_info(cluster_id=cluster_id)
            # cluster.add_block_related_info(block_id=block_id, debug=debug)
            data_obj.add_new_block_to_dict(block=self)
            data_obj.add_new_cluster_to_dict(cluster=cluster)


    # can i delete this func?? :/ check
    def center_block_image(self, debug=False, plot_images=False):
        debug_report(f"""
** running center_block_image for block{self.block_id}
min_max_coords_of_clusters={self.min_max_coords_of_clusters}', debug)
start x,y: {(self.start_x, self.start_y)}, backup start x,y: {(self.backup_start_x, self.backup_start_y)}
""", debug=debug)

        # first time called is from 'final_edits_after_adding_clusters_to_block' in 'connect_clusters_to_blocks
        ### no backup saving!
        # debug_report(f'going from center_block_image to reset_block_xy_definition and save_backup is {save_backup}',
        #              debug)
        # self.reset_block_xy_definition(
        #     start_coords=(new_start_x, new_start_y), end_coords=(new_end_x, new_end_y), debug=debug
        # )


    # B6
    def reset_min_max_coords_of_clusters(self, debug=False):
        self.min_max_coords_of_clusters = {'min_x': None, 'min_y': None, 'max_x': None, 'max_y': None}


    def add_cluster_related_info(self, cluster_id):
        if cluster_id not in self.clusters_ids_list:
            self.clusters_ids_list.append(cluster_id)
            # print(f'adding {cluster_id} info to block{self.block_id}')
        else:
            debug_report(f'cluster id {cluster_id} already exists in block{self.block_id},debug')

    # B2
    def restore_backup(self, restore_original_clusters_too=False, debug=False):
        debug_report(
            f"** running restore_backup for block{self.block_id} with restore_original_clusters_too={restore_original_clusters_too}",
            debug)
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        block_size = data_obj.block_size
        debug_report(
            f"starting clusters_ids_list: {self.clusters_ids_list}, start_x={self.start_x}, start_y={self.start_y}, dont_touch_this_block={self.dont_touch_this_block}",
            debug)

        if self.dont_touch_this_block:
            print(f'cancelled restore_backup for block{self.block_id}')
            return


        if self.start_x != self.backup_start_x or self.start_y != self.backup_start_y:
            debug_report(
                f"gonna restore (x,y) from backup -> right now: {self.start_x}-{self.end_x} & {self.start_y}-{self.end_y}",
                debug)
            new_start_x = deepcopy(self.backup_start_x)
            new_start_y = deepcopy(self.backup_start_y)
            new_end_x = self.start_x + int(block_size)
            new_end_y = self.start_y + int(block_size)
            debug_report(f'new start and end from backup: {new_start_x}-{new_end_x} & {new_start_y}-{new_end_y} ',
                         debug)
            self.reset_block_start_end_coords(start_coords=(new_start_x, new_start_y),
                                              end_coords=(new_end_x, new_end_y), debug=debug)

        else:
            debug_report(
                f"No need to restore the start x,y because it matches to backup: {(self.backup_start_x, self.backup_start_y)}",
                debug)

        for cluster_id in self.clusters_ids_list:
            if cluster_id not in self.backup_clusters_ids_list:
                data_obj.delete_cluster(cluster_id)
                debug_report(f"removed {cluster_id} from clusters_dict! :/", debug)
            elif restore_original_clusters_too:
                debug_report(f"restoring from ORIGINAL backup for cluster{cluster_id}", debug)
                ClassesFunctions.restore_cluster_from_backup(file_name=self.file_name, cluster_id=cluster_id, debug=debug)
            else:
                continue

        self.clusters_ids_list = deepcopy(self.backup_clusters_ids_list)
        debug_report(f"updated block{self.block_id}.clusters_ids_list to {self.clusters_ids_list}", debug)

        data_obj.add_new_block_to_dict(self)
        debug_report(f"-- done with restore_backup for block{self.block_id} -> {self.__dict__} \n", debug)

    def make_neg_fg_mask(self, fg_inc_pixels=2, margin_pixels=2, bg_r=4,
                         debug=False):  # neg fg mask is 0 when there is a spot fg or at margins.
        debug_report(f'** running make_neg_fg_mask for block{self.block_id} with clusters: {self.clusters_ids_list}',
                     debug)
        # checkme
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        # image = ScanDataObj.get_image_from_dict(file_name=self.file_name, dict_key='file_image')
        image = ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='image')
        neg_fg_mask = np.ones(image.shape)
        for cluster_id in self.clusters_ids_list:
            cluster = data_obj.get_cluster(cluster_id)
            for x, y, r in cluster.spots_coords_in_block_list:
                r += fg_inc_pixels
                cv2.circle(neg_fg_mask, (x, y), r + margin_pixels, 0, thickness=-1)
            debug_report(f"last circle of cluster{cluster_id} is at {(x, y, r)}", debug)

        # checkme
        ScanDataObj.add_block_related_images_to_dict(file_name=self.file_name, block_id=self.block_id,
                                                     image_tag='neg_fg_mask', image_file=neg_fg_mask)
        if debug:
            block_image = self.plot_block(plot_images=False, description='cAb:intensities')
            CommonFunctions.display_in_console(block_image, fig_size=[5,5])
            CommonFunctions.display_in_console(255 * neg_fg_mask, fig_size=[5,5])

        return neg_fg_mask

    def measure_all_fg_bg_of_block(self, fg_inc_pixels=2, margin_pixels=2,
                                   bg_r=4, debug=False, plot_images=False):
        debug_report(
            f'** running measure_all_fg_bg_of_block for block{self.block_id} with clusters: [{self.clusters_ids_list}]',
            debug)

        # checkme
        neg_fg_mask = self.make_neg_fg_mask(debug=debug)
        debug_report(f'neg_fg_mask is {neg_fg_mask.shape}', debug)
        data_obj = ScanDataObj.get_scan_data(file_name=self.file_name)
        for cluster_id in self.clusters_ids_list:
            cluster = data_obj.get_cluster(cluster_id)
            cluster.measure_all_fg_bg_of_a_cluster(
                neg_fg_mask=neg_fg_mask,
                block_image=ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='image'),
                debug=debug,
                fg_inc_pixels=fg_inc_pixels,
                margin_pixels=margin_pixels,
                bg_r=bg_r,
                plot_images=plot_images
            )
        self.fg_bg_calculated_flag = True
        debug_report(f'Done! -> F and B are calculated for clusters in Block{self.block_id}\n', debug)

    def calculate_results_in_block(self, sigma1=2, sigma2=3, debug=False, over_write=True):
        debug_report(f"**running calculate_results_in_block for {self.block_id}", debug)
        delete_lvl0 = 0
        delete_lvl1 = 0
        delete_lvl2 = 0
        total_points = 0

        #         if self.results_counts:
        #             debug_report(f'skipping all calculations because self.results_counts={self.results_counts}', debug)
        #             return self.results_counts
        data_obj = ScanDataObj.get_scan_data(file_name=self.file_name)
        for cluster_id in self.clusters_ids_list:
            cluster = data_obj.get_cluster(cluster_id)
            #             if cluster.short_clean_signal_list and not over_write:
            #                 debug_report(f'skipping cluster{cluster_id} because it has already been done!', debug)
            #                 continue
            cluster.signal_list = [fg - bg for fg, bg in zip(cluster.mean_fg_list, cluster.mean_bg_list)]
            debug_report(
                f"""working on cluster {cluster_id}: with fg={[int(x) for x in cluster.mean_fg_list]}, bg={[int(x) for x in cluster.mean_bg_list]} => signal = {[int(x) for x in cluster.signal_list]}""",
                debug)

            # first remove the negative values
            neg_values = list(np.where(np.array([x for x in cluster.signal_list]) < 0)[0])
            delete_lvl0 += len(neg_values)
            clean_signal_list = [x if x >= 0 else '' for x in cluster.signal_list]

            # then two levels of outlier removal (with two different sigma)
            no_outlier_lvl1, delete_indices_lvl1 = CommonFunctions.outlier_removal(input_list=clean_signal_list,
                                                                                   sigma=sigma1)
            deleted_count_lvl1 = len(delete_indices_lvl1)
            debug_report(f"outlier removal level 1: sigma={sigma1} and deleted {deleted_count_lvl1} spots", debug)

            no_outlier_lvl2, delete_indices_lvl2 = CommonFunctions.outlier_removal(input_list=no_outlier_lvl1,
                                                                                   sigma=sigma2)
            deleted_count_lvl2 = len(delete_indices_lvl2)
            debug_report(f"outlier removal level 2: sigma={sigma2} and deleted {deleted_count_lvl2} spots", debug)

            cluster.clean_signal_list = no_outlier_lvl2
            cluster.short_clean_signal_list = [num for num in no_outlier_lvl2 if num != '']
            if len(cluster.short_clean_signal_list) > 0:
                cluster.mean_cluster_signal = np.mean(cluster.short_clean_signal_list)
            debug_report(f"mean signal = {cluster.mean_cluster_signal}", debug)

            delete_lvl1 += delete_lvl1 + deleted_count_lvl1
            delete_lvl2 += delete_lvl2 + deleted_count_lvl2
            total_points += total_points + len(cluster.signal_list)
        self.results_counts = [delete_lvl1, delete_lvl2, total_points]
        return self.results_counts

    # B7
    def update_min_max_coords_of_clusters(self, coords_list=None, debug=False):
        # debug=True
        debug_report(f'** running update_min_max_coords_of_clusters for {self.block_id} & coords_list={coords_list}',debug)
        debug_report(f'current min_max_coords_of_clusters is {self.min_max_coords_of_clusters}', debug)

        if not coords_list: # when it's called from ClassesFunctions/initiating all blocks
            debug_report(f'going for these clusters: {self.clusters_ids_list}',debug)
            for cluster_id in self.clusters_ids_list:
                cluster = ScanDataObj.get_scan_data(self.file_name).get_cluster(cluster_id)
                debug_report(f'1) before update min_max is: {self.min_max_coords_of_clusters }',debug)
                debug_report(f'and input_coords_list = {cluster.spots_coords_list}',debug)
                self.min_max_coords_of_clusters = CommonFunctions.find_min_max_coords(
                    input_coords_list=cluster.spots_coords_list,
                    min_max_dict=self.min_max_coords_of_clusters
                )
                debug_report(f'2) after update min_max is: {self.min_max_coords_of_clusters }',debug)

        else:
            # self.reset_min_max_coords_of_clusters(debug=debug)
            debug_report(f'3) before update min_max is: {self.min_max_coords_of_clusters}',debug)
            debug_report(f'1and input_coords_list = {coords_list}', debug)
            self.min_max_coords_of_clusters = CommonFunctions.find_min_max_coords(
                input_coords_list=coords_list,
                min_max_dict=self.min_max_coords_of_clusters
            )
            debug_report(f'4) after update min_max is: {self.min_max_coords_of_clusters}',debug)

        debug_report(f'finally @ block{self.block_id} -> [min_x, max_x, min_y, max_y] = {self.min_max_coords_of_clusters}', debug)

        return

    def create_block_mask(self, debug=False, plot_images=False):
        # debug = True
        debug_report(f'** running create_block_mask for {self.block_id}', debug)
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        avg_spot_r = data_obj.avg_spot_r
        scan_size=data_obj.scan_size

        original_size = ScanDataObj.get_block_image(
            file_name=self.file_name,
            block_id=self.block_id,
            image_tag='image').shape
        debug_report(f'starting mask (original image) size = {original_size[::-1]}', debug)
        mask = np.zeros(original_size)

        # Finding min and max values of clusters (x,y) to find the corners of the mask:
        for cluster_id in self.clusters_ids_list:
            cluster = data_obj.get_cluster(cluster_id)
            coords_list = cluster.spots_coords_in_block_list
            for x, y, r in coords_list:
                cv2.circle(mask, (x, y), avg_spot_r, 255, thickness=-1)
        #             [min_x, max_x, min_y, max_y] = self.update_min_max_coords_of_clusters(self, coords_list, debug)

        mask = mask.astype(np.uint8)
        margin = int(3*data_obj.avg_spot_r) #checkme:
        debug_report(
            f'after the loop, mask is {mask.shape}, min_max_coords_of_clusters={self.min_max_coords_of_clusters}, start (x,y)={(self.start_x, self.start_y)} and margin = {margin}',
            debug)

        # changing absolute coords to relative to the block start coords!
        mask_min_x = int(max(0, self.min_max_coords_of_clusters['min_x'] - self.start_x - margin))
        mask_max_x = int(self.min_max_coords_of_clusters['max_x'] - self.start_x + margin)
        mask_min_y = int(max(0, self.min_max_coords_of_clusters['min_y'] - self.start_y - margin))
        mask_max_y = int(self.min_max_coords_of_clusters['max_y'] - self.start_y + margin)
        debug_report(
            f'mask_min_x={mask_min_x}, mask_max_x={mask_max_x}, mask_min_y={mask_min_y}, mask_max_y={mask_max_y}',
            debug)

        mask = mask[mask_min_y:mask_max_y, mask_min_x:mask_max_x]

        self.mask_start_coords = [mask_min_x, mask_min_y]
        self.mask_end_coords = [mask_max_x, mask_max_y]
        debug_report(f'Final mask shape is {mask.shape} and mask_start_coords={self.mask_start_coords}', debug)

        ScanDataObj.add_block_related_images_to_dict(file_name=self.file_name, block_id=self.block_id,
                                                     image_tag='block_mask', image_file=mask)
        if plot_images:
            self.plot_block_mask()

        #         self.center_block_image(debug=debug)
        return mask

    def plot_block_mask(self):
        mask = ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag={'block_mask'})
        CommonFunctions.display_in_console(mask, plot_images=1, fig_size=500)

    # B10
    def add_new_cluster_to_block_from_template(self, ref_cluster, displacement_vec=None, debug=False):
        debug_report(f'** running add_new_cluster_to_block_from_template for block{self.block_id} based on ref cluster{ref_cluster.cluster_id}', debug)

        if displacement_vec is None:
            displacement_vec = [0, 0, 0]

        debug_report(
            f'displacement_vec={displacement_vec} + self.start x,y=[{self.start_x, self.start_y}] - ref.start x,y: [{ref_cluster.block_start_x, ref_cluster.block_start_y}]',
            debug)
        new_cluster_id = ClassesFunctions.get_max_cluster_id(file_name=self.file_name) + 1
        new_cluster = ClassesFunctions.create_new_cluster(file_name=self.file_name, cluster_id=new_cluster_id, debug=debug)
        new_cluster.add_block_info(block_id=self.block_id, debug=debug)
        new_cluster.init_cluster_based_on_ref_outside_of_block(
            ref_cluster=ref_cluster,
            displacement_vec=displacement_vec,
            debug=debug)
        new_cluster.add_block_info(block_id=self.block_id, debug=debug)  # che konam?
        self.add_cluster_related_info(new_cluster_id)
        ScanDataObj.get_scan_data(self.file_name).add_new_cluster_to_dict(new_cluster)
        debug_report(
            f'final coords: {new_cluster.spots_coords_list} or IN BLOCK:{new_cluster.spots_coords_in_block_list}',
            debug)
        return new_cluster_id

    # B11
    # checkme double check! dont need to check both ways:/
    def check_for_rep_cluster_in_block(self, debug=False, debug_clusters=None,
                                       acceptable_distance=None, offsets_vec=None):
        #         debug=True
        debug_report(f'** running check_for_rep_cluster_in_block for block{self.block_id}', debug)

        if offsets_vec is None:
            offsets_vec = [0, 0]
        if acceptable_distance is None:
            acceptable_distance = [250, 20]
        if debug_clusters is None:
            debug_clusters = []

        data_obj = ScanDataObj.get_scan_data(self.file_name)
        debug_report(self.full_report(return_str=debug), debug)

        these_are_matched_already = []
        for ref_cluster_id in self.clusters_ids_list:
            debug_cluster_r = True if debug or ref_cluster_id in debug_clusters else False
            debug_report(f'** Working on ref_cluster{ref_cluster_id}, offsets_vec={offsets_vec}', debug_cluster_r)

            if ref_cluster_id in these_are_matched_already:
                continue

            ref_cluster = data_obj.get_cluster(ref_cluster_id)
            debug_report(f'ref_cluster: {ref_cluster.full_report(return_str=debug_cluster_r)}', debug_cluster_r)

            for second_cluster_id in self.clusters_ids_list:
                debug_cluster_s = True if debug or debug_cluster_r or second_cluster_id in debug_clusters else False
                if second_cluster_id <= ref_cluster_id:
                    debug_report(f'hehe these are the same! :D', debug_cluster_s)
                    continue

                if second_cluster_id in these_are_matched_already:
                    continue

                second_cluster = data_obj.get_cluster(second_cluster_id)
                debug_report(f'second_cluster: {second_cluster.full_report(return_str=debug_cluster_s)}', debug_cluster_s)

                avg_x_dist = np.abs(ref_cluster.avg_x_in_block - second_cluster.avg_x_in_block + offsets_vec[0])
                avg_y_dist = np.abs(ref_cluster.avg_y_in_block - second_cluster.avg_y_in_block + offsets_vec[1])

                debug_report(
                    f'avg_x_dist={avg_x_dist} and avg_y_dist={avg_y_dist} [acceptable_distance={acceptable_distance}]',
                    debug_cluster_s)

                if avg_x_dist > acceptable_distance[0] or avg_y_dist > acceptable_distance[1]:  # not a match!
                    debug_report(
                        f'ref_cluster{ref_cluster_id} & second_cluster{second_cluster_id} are not matched because dist={(avg_x_dist, avg_y_dist)}',
                        debug_cluster_s)
                    continue

                debug_report(f'MATCH! -> ref_cluster{ref_cluster_id} matches second_cluster{second_cluster_id}', debug_cluster_s)

                ref_cluster.merge_with_another_cluster(other_cluster_id=second_cluster_id,
                                                       offsets_vec=offsets_vec, debug=debug_cluster_s)
                self.update_min_max_coords_of_clusters(coords_list=second_cluster.spots_coords_list, debug=debug_cluster_s)
                ref_cluster.merged_with_another = True
                #                 delete_cluster(second_cluster_id,debug=debug)
                self.clusters_ids_list.remove(second_cluster_id)
                these_are_matched_already += [ref_cluster_id, second_cluster_id]

        debug_report(f"[After] block{self.block_id}.clusters_ids_list: {self.clusters_ids_list}", debug)

    def full_report(self, return_str=None):
        if return_str in [False, 0]:
            return
        skip_attr = ['file_name','row_number','col_number','Ag_conc','cAb_names','dAb_map_key','dAb_label',
                     'target','results_counts','fg_bg_calculated_flag','intensities_dict_list']
        output = '\n'
        for attr, value in self.__dict__.items():
            if attr not in skip_attr:
                output += f'\t{attr}: {value}\n'

        data_obj = ScanDataObj.get_scan_data(self.file_name)
        for cid in self.clusters_ids_list:
            c = data_obj.get_cluster(cid)
            output += c.full_report(return_str=return_str)
        if return_str in [True, 1]:
            return output
        else:
            print(output)

    # B13
    def find_exact_match_move(self, template_block, acceptable_distance=None,
                              offsets_vec=None, debug=False, debug_clusters=None):
        #         debug=True
        debug_report(
            f'** running find_exact_match_move for block{self.block_id} based on template_block{template_block.block_id}',
            debug)
        if offsets_vec is None:
            offsets_vec = [0, 0]
        if acceptable_distance is None:
            acceptable_distance = [200, 30]
        if debug_clusters is None:
            debug_clusters = []

        data_obj = ScanDataObj.get_scan_data(self.file_name)
        # colored_image = CommonFunctions.make_3D_image(
        #     ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='log_image')
        # )
        init_max = deepcopy(ClassesFunctions.get_max_cluster_id(file_name=self.file_name))
        these_are_matched_already = []
        move_vec_list = []

        debug_report(f'acceptable_distance={acceptable_distance}, offsets_vec={offsets_vec}', debug)

        for cluster_id in template_block.clusters_ids_list:
            debug_cluster = True if debug or cluster_id in debug_clusters else False
            debug_report(f'-> Working on cluster{cluster_id} from the template...', debug_cluster)

            found_the_match = False
            cluster = data_obj.get_cluster(cluster_id)

            for self_cluster_id in self.clusters_ids_list:
                if found_the_match:
                    continue

                if self_cluster_id > init_max:
                    print(f'\n\n\n\t\t\t self_cluster_id={self_cluster_id} and init_max={init_max}\n\n\n')
                    continue

                if self_cluster_id in these_are_matched_already:
                    continue

                self_cluster = data_obj.get_cluster(self_cluster_id)

                avg_x_dist = np.abs(cluster.avg_x_in_block - self_cluster.avg_x_in_block + offsets_vec[0])
                avg_y_dist = np.abs(cluster.avg_y_in_block - self_cluster.avg_y_in_block + offsets_vec[1])


                if avg_x_dist > acceptable_distance[0] or avg_y_dist > acceptable_distance[1]:  # not a match!
#                     debug_report(f'cluster{cluster_id} & self_cluster{self_cluster_id} are not matched because dist={(avg_x_dist,avg_y_dist)}', debug)
                    continue

                found_the_match = True
                debug_report(
                    f'MATCH! -> cluster{cluster_id} matches self_cluster{self_cluster_id}... offsets_vec={offsets_vec}',
                    debug_cluster)

                this_move = self_cluster.get_avg_distance_to_another_cluster(
                    other_cluster_id=cluster_id,
                    offsets_vec=offsets_vec,
                    debug=debug_cluster
                )
                move_vec_list.append(this_move)
                debug_report(f'appended: {this_move} to move_vec_list', debug_cluster)

            if not found_the_match:
                debug_report(f'cluster{cluster_id} had no match here. continuing :D', debug)
                # found_the_match = True

        debug_report(f'move_vec_list = {move_vec_list}', debug)

        if not move_vec_list:
            debug_report(f'block{self.block_id}: exact_move={None}', debug)
            return None
        exact_move = np.mean(move_vec_list, axis=0).astype('int16')
        debug_report(f'block{self.block_id}: exact_move={exact_move}', debug)
        return exact_move

    # B8
    def add_clusters_from_another_block_template(self, template_block_id, acceptable_distance=None,
                                                 debug=False, plot_images=False, debug_clusters=None):
        debug_report(
            f'** B8: running add_clusters_from_another_block_template for block{self.block_id} based on template_block{template_block_id}',
            debug)
        if acceptable_distance is None:
            acceptable_distance = [200, 20]
        if debug_clusters is None:
            debug_clusters = []

        debug_report(f"[BEFORE] --> block{self.block_id}.clusters_ids_list = {self.clusters_ids_list}", debug)

        data_obj = ScanDataObj.get_scan_data(self.file_name)
        template_block = data_obj.get_block(template_block_id)
        exact_move = self.find_exact_match_move(template_block=template_block, debug=debug,
                                                debug_clusters=debug_clusters, acceptable_distance=acceptable_distance)
        debug_report(f"exact_move = {exact_move}", debug)
        offsets_vec = exact_move if exact_move is not None else [0, 0]

        # colored_image = CommonFunctions.make_3D_image(
        #     ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='log_image')
        # )
        init_max = deepcopy(ClassesFunctions.get_max_cluster_id(file_name=self.file_name))
        these_are_matched_already = []

        debug_report(f'starting adding clusters: acceptable_distance={acceptable_distance}, offsets_vec={offsets_vec}',
                     debug)

        for cluster_id in template_block.clusters_ids_list:
            debug_cluster = True if debug or cluster_id in debug_clusters else False
            debug_report(f'-> Working on cluster{cluster_id} from the template...', debug_cluster)

            found_the_match = False
            cluster = data_obj.get_cluster(cluster_id)

            for self_cluster_id in self.clusters_ids_list:
                debug_cluster = True if debug or debug_cluster or self_cluster_id in debug_clusters else False

                if found_the_match:
                    continue

                if self_cluster_id > init_max:
                    continue

                if self_cluster_id in these_are_matched_already:
                    continue

                self_cluster = data_obj.get_cluster(self_cluster_id)

                avg_x_dist = np.abs(cluster.avg_x_in_block - self_cluster.avg_x_in_block + offsets_vec[0])
                avg_y_dist = np.abs(cluster.avg_y_in_block - self_cluster.avg_y_in_block + offsets_vec[1])

                debug_report(
                    f'template_cluster{cluster_id}: avg_x_in_block={cluster.avg_x_in_block} and avg_y_in_block={cluster.avg_y_in_block}',
                    debug_cluster)
                debug_report(
                    f'self_cluster{self_cluster_id}: avg_x_in_block={self_cluster.avg_x_in_block} and avg_y_in_block={self_cluster.avg_y_in_block}',
                    debug_cluster)
                debug_report(f'avg_x_dist={avg_x_dist} and avg_y_dist={avg_y_dist}', debug_cluster)

                if avg_x_dist > acceptable_distance[0] or avg_y_dist > acceptable_distance[1]:  # not a match!
                    debug_report(
                        f'cluster{cluster_id} & self_cluster{self_cluster_id} are not matched because dist={(avg_x_dist, avg_y_dist)}',
                        debug_cluster)
                    continue

                debug_report(
                    f'MATCH! -> cluster{cluster_id} matches self_cluster{self_cluster_id}... offsets_vec={offsets_vec}',
                    debug_cluster)
                found_the_match = True
                self_cluster.merge_with_another_cluster(other_cluster_id=cluster_id, offsets_vec=offsets_vec, debug=debug_cluster)
                self_cluster.merged_with_another = True
                self.update_min_max_coords_of_clusters(coords_list=self_cluster.spots_coords_list, debug=debug_cluster)

            if not found_the_match:
                debug_report(f'cluster{cluster_id} had no match here. Will create a new cluster now....', debug)
                new_cluster_id = self.add_new_cluster_to_block_from_template(
                    debug=debug,
                    ref_cluster=cluster,
                    displacement_vec=[offsets_vec[0], offsets_vec[1], 0]  # to match the template on new block
                )
                debug_cluster = True if debug or new_cluster_id in debug_clusters else False

                debug_report(f'added the new cluster{new_cluster_id} to block{self.block_id}', debug_cluster)
                # found_the_match = True
                self.update_min_max_coords_of_clusters(
                    coords_list=data_obj.get_cluster(new_cluster_id).spots_coords_list,
                    debug=debug_cluster
                )

        debug_report(f"[After] block{self.block_id}.clusters_ids_list: {self.clusters_ids_list}", debug)
        if plot_images:
            print(f'block{self.block_id}:', end='')
            self.plot_block(debug=debug)

    def find_pattern_in_a_block(self, pattern, matching_method=cv2.TM_CCOEFF_NORMED,
                                preprocess_params=None, move_match=None, debug=False, plot_images=False):
        #         debug=True
        if move_match is None:
            move_match = [0, 0]
        debug_report(
            f'** running find_pattern_in_a_block for block{self.block_id} --- input move_match={move_match}',
            debug)

        image = ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='image')
        processed_image = CommonFunctions.image_preprocessing(input_image=image, params=preprocess_params)
        debug_report(f'processed_image is {processed_image.shape} and pattern is {pattern.shape}', debug)

        if plot_images:
            print(f'\nthis is the pre-processed image of block{self.block_id}',end='')
            CommonFunctions.display_in_console(processed_image, fig_size=[5,5])
            print(f'\nand this is the template pattern:', end='')
            CommonFunctions.display_in_console(pattern, fig_size=[5,5])

        match_result = cv2.matchTemplate(processed_image, pattern, matching_method)
        _, _, min_loc, max_loc = cv2.minMaxLoc(match_result)

        if matching_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            match_top_left = min_loc
        else:
            match_top_left = max_loc

        debug_report(f'match_top_left={match_top_left}', debug)
        match_top_left = [match + move for match, move in zip(match_top_left, move_match)]
        debug_report(f'moved the to match_top_left to this new location, based on move_match ({move_match}) --> match_top_left={match_top_left}', debug)

        if plot_images or debug:
            pattern_height, pattern_width = pattern.shape[:2]
            debug_report(f'pattern_height, pattern_width={pattern.shape[:2]}', debug)

            match_bottom_right = (match_top_left[0] + pattern_width, match_top_left[1] + pattern_height)

            print('this is the processed image + match box!')
            cv2.rectangle(processed_image, match_top_left, match_bottom_right, color=(255, 255, 255), thickness=5)
            CommonFunctions.display_in_console(processed_image, fig_size=[5,5])
        return match_top_left

    # B9
    def move_block_based_on_template_mask(self, template_block_id, matching_method=cv2.TM_CCOEFF_NORMED,
                                          preprocess_params=None, debug=False,
                                          plot_images=False, move_match=None):
        debug_report(f'** running move_block_based_on_template_mask for block{self.block_id}, '
                     f'based on template block{template_block_id} --- input move_match={move_match}', debug)
        if move_match is None:
            move_match = [0, 0]
        data_obj = ScanDataObj.get_scan_data(file_name=self.file_name)
        block_size = data_obj.block_size
        template_block = data_obj.get_block(template_block_id)
        template_block_mask = ScanDataObj.get_block_image(
            file_name=self.file_name,
            block_id=template_block.block_id,
            image_tag='block_mask'
        )

        # if move_match != [0,0]:
        #     self.set_start_and_end_of_block()
        match_top_left = self.find_pattern_in_a_block(
            pattern=template_block_mask,
            matching_method=matching_method,
            preprocess_params=preprocess_params,
            debug=debug,
            plot_images=plot_images,
            move_match=move_match
        )

        cluster_move = move_match

        delta_x = match_top_left[0] - template_block.mask_start_coords[0]
        delta_y = match_top_left[1] - template_block.mask_start_coords[1]

        debug_report(
            f'(start_x,start_y): {(self.start_x, self.start_y)}, match_top_left: {match_top_left}, template_block.mask_start_coords: {template_block.mask_start_coords} => delta x,y: {(delta_x, delta_y)}',
            debug)

        new_start_x = max(self.start_x + delta_x, 0)
        new_start_y = max(self.start_y + delta_y, 0)

        new_end_x = new_start_x + int(block_size)
        new_end_y = new_start_y + int(block_size)

        debug_report(
            f'(new_start_x,new_end_x): {(new_start_x, new_end_x)} & with size={block_size} => (new_start_y,new_end_y): {(new_start_y, new_end_y)}',
            debug)

        self.reset_block_start_end_coords(
            start_coords=(deepcopy(new_start_x), deepcopy(new_start_y)),
            end_coords=(deepcopy(new_end_x), deepcopy(new_end_y)),
            debug=debug,
        )

        if self.start_x + delta_x < 0 or self.start_y + delta_y < 0:
            cluster_move = [move_match[0] + delta_x, move_match[1] + delta_y]

        debug_report(f'move_match={move_match},delta_x_y={[delta_x, delta_y]},cluster_move={cluster_move}', debug)
        return {'move_match': move_match, 'cluster_move': cluster_move}

    # B12
    def create_clusters_from_another_block(self, template_block_id, matching_method=cv2.TM_CCOEFF_NORMED,
                                           debug_clusters=None, preprocess_params=None, debug=False,
                                           plot_images=False, move_match=None):
        #         debug=True
        debug_report(f'** running create_clusters_from_another_block from {template_block_id} in {self.block_id} '
                     f'--- input move_match={move_match}', debug)

        if move_match is None:
            move_match = [0, 0]
        if debug_clusters is None:
            debug_clusters = []

        acceptable_distance = [200, 30]

        data_obj = ScanDataObj.get_scan_data(file_name=self.file_name)
        template_block = data_obj.get_block(template_block_id)
        debug_report(f'move_match={move_match} and preprocess_params={preprocess_params}', debug)
        debug_report(f'this is the template_block:{template_block.full_report(return_str=debug)}', debug)

        move_results = self.move_block_based_on_template_mask(
            template_block_id,
            matching_method=matching_method,
            preprocess_params=preprocess_params,
            debug=debug,
            plot_images=plot_images,
            move_match=move_match
        )

        move_match = move_results['move_match']
        cluster_move = move_results['cluster_move']
        debug_report(f'move_match={move_match}, cluster_move={cluster_move}',debug)

        self.check_for_rep_cluster_in_block(
            offsets_vec=move_match,
            debug=debug,
            debug_clusters=debug_clusters,
            acceptable_distance=acceptable_distance
        )

        self.add_clusters_from_another_block_template(
            template_block_id=template_block_id,
            debug=debug,
            plot_images=plot_images,
            debug_clusters=debug_clusters,
            acceptable_distance=acceptable_distance
        )



    # todo: fix the two columns shit :/
    def add_names_to_clusters(self, two_columns=True, debug=False):
        #         debug=True
        debug_report(f'** running add_names_to_clusters for block{self.block_id} with names: {self.cAb_names}', debug)
        data_obj = ScanDataObj.get_scan_data(file_name=self.file_name)

        #         if not two_columns:
        #             print('shit!')
        #             return

        #         self.sorted_clusters_ids_list = []
        #         self.cAb_names = deepcopy(cAb_names_vec)

        self.cAb_names = deepcopy(data_obj.cAb_names)

        if len(self.cAb_names) < len(self.clusters_ids_list):
            diff = len(self.clusters_ids_list) - len(self.cAb_names)
            data_obj.cAb_names += [0] * diff
            debug_report(f'had to add {diff} zeros at the end of cAb_names!!!', debug=True)

        debug_report(
            f'block{self.block_id} has {len(self.clusters_ids_list)} clusters and {len(self.cAb_names)} names for them',
            debug)
        self.sorted_clusters_ids_list = []
        group1, group2 = [], []
        for cluster_id in self.clusters_ids_list:
            cluster = data_obj.get_cluster(cluster_id)
            if cluster.position_ind == 0:
                group1.append((cluster_id, cluster.avg_x_in_block, cluster.avg_y_in_block))
                debug_report(f'cluster{cluster_id} at {(cluster.avg_x_in_block, cluster.avg_y_in_block)} is in group1',
                             debug)

            else:
                group2.append((cluster_id, cluster.avg_x_in_block, cluster.avg_y_in_block))
                debug_report(f'cluster{cluster_id} at {(cluster.avg_x_in_block, cluster.avg_y_in_block)} is in group2',
                             debug)

        # Sort each group based on average y value
        group1_sorted = sorted(group1, key=lambda x: x[2])
        group2_sorted = sorted(group2, key=lambda x: x[2])

        group1_ids = [x[0] for x in group1_sorted]
        group2_ids = [x[0] for x in group2_sorted]
        debug_report(f'group1={group1_ids}, group2={group2_ids}', debug)

        i = 0
        for cluster_id in group1_ids:
            cluster = data_obj.get_cluster(cluster_id)
            cluster.name = data_obj.cAb_names[i]
            i += 1
            debug_report(f'cluster{cluster_id} -> i = {i} and name = {cluster.name}', debug)
            self.sorted_clusters_ids_list.append(cluster_id)

        for cluster_id in group2_ids:
            cluster = data_obj.get_cluster(cluster_id)
            cluster.name = data_obj.cAb_names[i]
            i += 1
            debug_report(f'cluster{cluster_id} -> i = {i} and name = {cluster.name}', debug)
            self.sorted_clusters_ids_list.append(cluster_id)
        debug_report(f'final cluster names in this block -> {self.cAb_names}', debug)

    def plot_block(self, plot_images=True, debug=False, label=True, fig_size=None, crop_to_mask=False,
                   description='cluster_ids', with_border=False, custom_cAb_names=None, crop_size=1):
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        avg_spot_r = data_obj.avg_spot_r
        # scan_size = data_obj.scan_size
        scaled_image = ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='scaled_image')
        colored_image = CommonFunctions.make_3D_image(scaled_image)

        debug_report(f'there are {len(self.clusters_ids_list)} clusters in block {self.block_id}', debug)

        i = 0
        for cluster_id in self.clusters_ids_list:
            debug_report(f'Loading cluster{cluster_id}:', debug)
            cluster = data_obj.get_cluster(cluster_id)
            debug_report(f'its dict: {cluster.__dict__}', debug)

            intensity = '-'
            if cluster.mean_cluster_signal:
                intensity = int(cluster.mean_cluster_signal)
            if description == 'cluster_ids':
                cluster.plot_cluster_on_image(colored_image, debug=debug)
            elif description == 'cAb_names':
                if not self.cAb_names:
                    self.add_names_to_clusters(debug=debug)
                cluster.plot_cluster_on_image(colored_image, name_tag=cluster.name, debug=debug)
            elif description == 'intensities':
                cluster.plot_cluster_on_image(colored_image, name_tag=intensity, debug=debug)
            elif description == 'cAb:intensities':
                cluster.plot_cluster_on_image(colored_image, name_tag=f'{cluster.name}:{intensity}', debug=debug)
            elif description == 'clusters_and_mask_border':
                margin = avg_spot_r + 5
                start_point = (self.min_max_coords_of_clusters['min_x'] - margin - self.start_x,
                               self.min_max_coords_of_clusters['min_y'] - margin - self.start_y)
                end_point = (self.min_max_coords_of_clusters['max_x'] + margin - self.start_x,
                             self.min_max_coords_of_clusters['max_y'] + margin - self.start_y)
                debug_report(f'for mask border, start_point={start_point} and end_point={end_point}', debug)
                colored_image = cv2.rectangle(colored_image, start_point, end_point, (70, 70, 70), 2)
                cluster.plot_cluster_on_image(colored_image, debug=debug)

            elif description == 'custom':
                cluster.plot_cluster_on_image(colored_image, name_tag=custom_cAb_names[i], debug=debug)
            else:
                print('use one of these dscr: [with_cluster_ids, with_cAb_names, cAb:intensities]')
            i += 1

        if with_border:
            self.add_border_to_block_image(colored_image, debug)

        cropped_img = colored_image[crop_size:-crop_size, crop_size:-crop_size]

        if crop_to_mask:
            debug_report(f'cropping colored image {colored_image.shape} based on mask_start_coords={self.mask_start_coords} & mask_end_coords={self.mask_end_coords}', debug)
            pad = 60
            cropped_img = colored_image[max(0,self.mask_start_coords[1]-pad):self.mask_end_coords[1]+pad,
                          max(0,self.mask_start_coords[0]-pad):self.mask_end_coords[0]+pad]
        debug_report(f'colored_image is {colored_image.shape} and cropped_img is {cropped_img.shape}', debug)

        if label is None:
            label = False

        elif label is True:
            label = self.block_id
        cv2.putText(cropped_img, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        coords = (int(self.start_x), int(self.start_y))
        if debug:
            cv2.putText(cropped_img, f'{coords}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        CommonFunctions.display_in_console(image=cropped_img, plot_images=plot_images, fig_size=fig_size)
        if not plot_images:
            return cropped_img

    # B3
    # todo: checkme here! the for loop can just call a function in cluster...
    def edit_spots_manually(self, manual_spot_edit_dict, debug=False, debug_clusters=None):
        #         debug=True
        debug_report(f'** running edit_spots_manually for block{self.block_id} with command: {manual_spot_edit_dict}',
                     debug)
        if debug_clusters is None:
            debug_clusters = self.clusters_ids_list

        data_obj = ScanDataObj.get_scan_data(self.file_name)
        # move_all_direction = None
        for cluster_id, commands_list in manual_spot_edit_dict.items():
            debug_cluster = True if debug or cluster_id in debug_clusters else False
            # if cluster_id == self.block_id:
            #     move_all_direction = commands_list
            #     continue
            if cluster_id not in self.clusters_ids_list:
                continue

            cluster = data_obj.get_cluster(cluster_id)
            debug_report(f'gonna edit cluster{cluster_id} with commands_list={commands_list}', debug_cluster)

            # if commands_list[0] == 'restore':
            #     cluster.restore_coords_backup_and_more(debug=debug)
            #     commands_list = commands_list[1:]
            #     debug_report(f'just restored cluster{cluster_id} to backup.', debug)

            # if move_all_direction:
            #     commands_list = [move_all_direction] + commands_list
            cluster.manual_edit_spots_in_cluster(commands_list=commands_list, debug=debug_cluster)

        self.reset_min_max_coords_of_clusters()
        self.update_min_max_coords_of_clusters(debug=debug)

    def give_image_from_backup(self, debug=False):
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        block_size = data_obj.block_size
        original_image = ScanDataObj.get_image_from_dict(file_name=self.file_name, dict_key='file_image')
        original_log_image = ScanDataObj.get_image_from_dict(file_name=self.file_name, dict_key='file_scaled_image')

        #         debug=True
        x_start = self.backup_start_x
        y_start = self.backup_start_y
        debug_report(f'backup start: {[self.backup_start_x, self.backup_start_y]} and avg_width={block_size}',
                     debug)
        before_image = original_image[y_start:y_start + int(block_size), x_start:x_start + int(block_size)]
        before_log_image = original_log_image[y_start:y_start + int(block_size),
                           x_start:x_start + int(block_size)]
        #         before_image = load_image_from_images_dict(block_id=self.block_id, label='backup_image', debug=debug)
        #         before_log_image = load_image_from_images_dict(block_id=self.block_id, label='backup_log_image', debug=debug)
        return before_image, before_log_image

    def add_border_to_block_image(self, input_image, debug=False):
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        margin = data_obj.avg_spot_r + 5
        start_point = (self.min_max_coords_of_clusters['min_x'] - margin - self.start_x,
                       self.min_max_coords_of_clusters['min_y'] - margin - self.start_y)
        end_point = (self.min_max_coords_of_clusters['max_x'] + margin - self.start_x,
                     self.min_max_coords_of_clusters['max_y'] + margin - self.start_y)
        input_image = cv2.rectangle(input_image, start_point, end_point, (70, 70, 70), 2)
        debug_report(f'-> for mask border, start_point={start_point} and end_point={end_point}', debug)

    #         return input_image

    def plot_before_after_of_block(self, debug=False):
        #         debug=True

        scaled_image = ScanDataObj.get_block_image(file_name=self.file_name,block_id=self.block_id, image_tag='scaled_image')
        after = CommonFunctions.make_3D_image(scaled_image)

        _, before_log_image = self.give_image_from_backup(debug=debug)
        before = CommonFunctions.make_3D_image(before_log_image)

        data_obj = ScanDataObj.get_scan_data(file_name=self.file_name)
        self.add_border_to_block_image(after, debug)
        #         margin = global_avg_r+5
        #         start_point = (self.min_max_coords_of_clusters['min_x']-margin - self.start_x,self.min_max_coords_of_clusters['min_y']-margin - self.start_y)
        #         end_point = (self.min_max_coords_of_clusters['max_x']+margin - self.start_x,self.min_max_coords_of_clusters['max_y']+margin - self.start_y)
        #         after = cv2.rectangle(after, start_point, end_point, (70,70,70), 2)
        #         debug_report(f'-> for mask border, start_point={start_point} and end_point={end_point}', debug)

        debug_report(f'before image is {before.shape} and after image is {after.shape}', debug)
        for cluster_id in self.clusters_ids_list:
            cluster = data_obj.get_cluster(cluster_id)
            debug_report(f'this is the current cluster: {cluster.full_report(return_str=debug)}', debug)

            debug_report(f'plotting cluster{cluster_id} on the "after" image, with coords: {cluster.spots_coords_list}',
                         debug)
            cluster.plot_cluster_on_image(after, debug=debug, size=4)
            original_cluster = data_obj.get_cluster_backup(cluster_id)
            debug_report(f'this is the backup cluster: {original_cluster.full_report(return_str=debug)}', debug)

            if original_cluster:
                debug_report(
                    f'plotting the original cluster{original_cluster.cluster_id} on the "before" image, with coords: {original_cluster.spots_coords_list}',
                    debug)
                #                 displacement = [-x for x in self.mask_offsets]
                displacement = [0, 0]
                original_cluster.plot_cluster_on_image(before, displacement=displacement, debug=debug, size=4)

        cv2.putText(before, 'Before', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(after, 'After', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        if debug:
            after_start = (int(self.start_x), int(self.start_y))
            before_start = (int(self.backup_start_x), self.backup_start_y)
            #             after_end = (self.start_x, self.start_y)
            #             before_end = (self.backup_start_x, self.backup_start_y)

            debug_report(f'after_start={after_start} and before_start={before_start}', debug)
            cv2.putText(after, str(after_start), (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(before, str(before_start), (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        crop = 1
        #         crop = 200
        cropped_after = after[crop:-crop, crop:-crop]
        debug_report(f'crop={crop}', debug)

        combined_image = CommonFunctions.pad_and_concat_images(before, cropped_after)
        CommonFunctions.display_in_console(combined_image, text=f'block{self.block_id}: before & after')

    #     def move_block_match(self,move_block_match=[0,0], debug=False):
    #         self.start_x += move_block_match[0]
    #         self.start_y += move_block_match[1]

    # B1
    def edit_block(self, debug=False, with_restore=False, plot_before_after=True,
                   manual_spot_edit_dict=None, overwrite=False, debug_clusters=None):
        if manual_spot_edit_dict is None:
            manual_spot_edit_dict = {}
        if debug_clusters is None:
            debug_clusters = {}

        if overwrite:
            self.dont_touch_this_block = False
        if self.dont_touch_this_block:
            print(f"will not be editing block{self.block_id}...")
            return

        if with_restore:
            self.restore_backup(restore_original_clusters_too=True, debug=debug)

        self.edit_spots_manually(manual_spot_edit_dict=manual_spot_edit_dict,debug=debug, debug_clusters=debug_clusters)
        self.update_block_start_end_from_clusters_min_max(debug=debug)
        # self.reset_min_max_coords_of_clusters(debug=debug) # changed this...need to check
        # self.center_block_image(debug=debug)
        self.plot_before_after_of_block(debug=debug) if plot_before_after else None


    def give_block_data_df(self, debug=False, over_write=False, fg_inc_pixels=2, margin_pixels=2, bg_r=4,
                           image_size=300):
        debug_report(f'** running give_block_data_df for block{self.block_id} (over_write={over_write})', debug)

        columns = ['Block_ID', 'Col', 'Row', 'Ag_Conc.', 'target', 'dAb_name', 'cAb_name', 'Cluster_ID',
                   'Spot_Index', 'F', 'B', 'F_B', 'F_B_PostProcess', 'Average_F_B_PostProcess']
        #         columns = ['Block_ID', 'Col','Row', 'cAb_name', 'Cluster_ID', 'Spot_Index',
        #                    'F', 'B', 'F_B', 'F_B_PostProcess', 'Average_F_B_PostProcess']

        #         if self.intensities_dict_list and not over_write:
        #             debug_report(f'gonna skip everythin here because self.intensities_dict_list is not None', debug)
        #             return self.intensities_dict_list, columns
        data_obj = ScanDataObj.get_scan_data(file_name=self.file_name)
        debug_report(f'1) self.fg_bg_calculated_flag={self.fg_bg_calculated_flag}', debug)
        if not self.fg_bg_calculated_flag:
            self.measure_all_fg_bg_of_block(debug=debug, fg_inc_pixels=fg_inc_pixels,
                                            margin_pixels=margin_pixels, bg_r=bg_r)

        debug_report(f'2) self.results_counts={self.results_counts}', debug)
        if not self.results_counts:
            self.calculate_results_in_block(debug=debug)

        intensities_dict_list = []
        if not self.sorted_clusters_ids_list:
            self.add_names_to_clusters(debug=debug)

        for cluster_id in self.sorted_clusters_ids_list:
            if cluster_id not in self.clusters_ids_list:
                debug_report(f'!!@@!! cluster{cluster_id} is in sorted_clusters_ids_list but not clusters_ids_list',
                             debug)  # TODO
                continue
            cluster = data_obj.get_cluster(cluster_id)
            n_spots = len(cluster.spots_coords_list)
            debug_report(
                f'working on cluster{cluster_id}({cluster.name}) with {n_spots}spots -> FG:{cluster.mean_fg_list}, BG:{cluster.mean_bg_list}',
                debug)

            for i in range(n_spots):
                spot_data = {
                    'Block_ID': self.block_id,
                    'Ag_Conc.': self.Ag_conc,
                    'dAb_name': self.dAb_label,
                    'cAb_name': cluster.name,
                    'Cluster_ID': int(cluster.cluster_id),
                    'Spot_Index': i,
                    'F': cluster.mean_fg_list[i].astype('uint16'),
                    'B': cluster.mean_bg_list[i].astype('uint16'),
                    'F_B': cluster.mean_fg_list[i] - cluster.mean_bg_list[i].astype('uint16'),
                    'F_B_PostProcess': cluster.clean_signal_list[i],
                    'Average_F_B_PostProcess': cluster.mean_cluster_signal,
                    'target': self.target,
                    'Col': self.col_number,
                    'Row': self.row_number
                }
                debug_report(spot_data, debug)
                if i != 0:
                    spot_data['Average F-B PostProcess'] = ''
                intensities_dict_list.append(spot_data)

        block_intensities_df = pd.DataFrame(intensities_dict_list, columns=columns)

        block_intensities_df['Block_ID'] = block_intensities_df['Block_ID'].astype('string')
        block_intensities_df['Ag_Conc.'] = block_intensities_df['Ag_Conc.'].astype('float')
        block_intensities_df['dAb_name'] = block_intensities_df['dAb_name'].astype('string')
        block_intensities_df['cAb_name'] = block_intensities_df['cAb_name'].astype('string')
        block_intensities_df['target'] = block_intensities_df['target'].astype('string')
        block_intensities_df['Cluster_ID'] = block_intensities_df['Cluster_ID'].astype('int')
        block_intensities_df['Spot_Index'] = block_intensities_df['Spot_Index'].astype('int')
        block_intensities_df['Col'] = block_intensities_df['Col'].astype('int')
        block_intensities_df['Col'] = block_intensities_df['Col'].astype('int')
        block_intensities_df['F'] = block_intensities_df['F'].round().astype('int')
        block_intensities_df['B'] = block_intensities_df['B'].round().astype('int')
        block_intensities_df['F_B'] = block_intensities_df['F_B'].astype('int', errors='ignore')

        # block_intensities_df['F_B_PostProcess'] = block_intensities_df['F_B_PostProcess'].replace('', np.nan)
        block_intensities_df['F_B_PostProcess'] = block_intensities_df['F_B_PostProcess'].replace('',
                                                                                                  np.nan).infer_objects(
            copy=False)

        block_intensities_df['F_B_PostProcess'] = block_intensities_df['F_B_PostProcess'].astype('float').astype(
            pd.UInt16Dtype(), errors='ignore')

        block_intensities_df['Average_F_B_PostProcess'] = block_intensities_df['Average_F_B_PostProcess'].replace('',
                                                                                                                  np.nan)
        block_intensities_df['Average_F_B_PostProcess'] = block_intensities_df['Average_F_B_PostProcess'].astype(
            'float').astype(pd.UInt16Dtype(), errors='ignore')

        #         block_intensities_df['F_B_PostProcess'] = block_intensities_df['F_B_PostProcess'].astype(pd.UInt16Dtype())
        #         block_intensities_df['Average_F_B_PostProcess'] = block_intensities_df['Average_F_B_PostProcess'].astype(pd.UInt16Dtype())
        return block_intensities_df

#         self.intensities_dict_list = intensities_dict_list
#         return self.intensities_dict_list, columns

        if not results_columns_names:
            self.results_columns_names = ['Block_ID', 'Row', 'Col', 'cAb_name', 'Cluster_ID', 'Spot_Index',
                   'F', 'B', 'F_B', 'F_B_PostProcess', 'Average_F_B_PostProcess']