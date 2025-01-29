import numpy as np
import cv2
from copy import deepcopy
from scipy import ndimage
import traceback
from importlib import reload
from Functions import CommonFunctions, ClassesFunctions
from Classes import ScanDataObj
reload(ScanDataObj)
reload(ClassesFunctions)
reload(CommonFunctions)
from Functions.CommonFunctions import debug_report


class Cluster:
    def __init__(self, cluster_id,  file_name, name=None, spots_coords_list=None, spots_coords_in_block_list=None,
                 avg_x=None, avg_y=None, avg_r=None, avg_x_in_block=None, min_max_abs_coords=None, avg_y_in_block=None,
                 block_id=None, block_start_x=None, block_start_y=None, mean_fg_list=None, mean_bg_list=None,
                 backup_coords_list=None, backup_block_coords_list=None, signal_list=None, color=None,
                 clean_signal_list=None, short_clean_signal_list=None, mean_cluster_signal=None, position_ind=None,
                 already_restored=False, merged_with_another=False):

        self.cluster_id = cluster_id
        self.file_name = file_name

        if spots_coords_list is None:
            spots_coords_list = []
        if spots_coords_in_block_list is None:
            spots_coords_in_block_list = []

        self.spots_coords_list = spots_coords_list
        self.spots_coords_in_block_list = spots_coords_in_block_list
        self.avg_x = avg_x
        self.avg_y = avg_y
        self.avg_x_in_block = avg_x_in_block
        self.avg_y_in_block = avg_y_in_block
        self.avg_r = avg_r
        self.block_id = block_id
        self.block_start_x = block_start_x
        self.block_start_y = block_start_y
        self.mean_fg_list = mean_fg_list
        self.mean_bg_list = mean_bg_list
        self.backup_coords_list = backup_coords_list
        self.backup_block_coords_list = backup_block_coords_list
        self.signal_list = signal_list
        self.clean_signal_list = clean_signal_list
        self.short_clean_signal_list = short_clean_signal_list
        self.mean_cluster_signal = mean_cluster_signal
        self.color = color
        self.name = name
        self.position_ind = position_ind  # 0:first column of cAbs in the block, 1: second column of cAbs in the block, ...
        self.already_restored = already_restored
        self.merged_with_another = merged_with_another
        if min_max_abs_coords is None: # these are absolute values of coords (not in  block)
            self.min_max_abs_coords = {'min_x': None, 'max_x': None, 'min_y': None, 'max_y': None}

    def add_spot(self, spot_coords, debug=False):
        self.spots_coords_list.append([int(x) for x in spot_coords]) # no back has been saved yet.

    def add_new_spot_to_cluster(self, spot_coords, debug=False):
        debug_report(f'** running "add_new_spot_to_cluster" for cluster{self.cluster_id} -> spot_info: {spot_coords}',
                     debug)
        if not self.spots_coords_list:
            self.add_spot(spot_coords=spot_coords, debug=debug) # no backup has been saved yet.
            return

        # if there are other spots there already, need to check if the spot is a duplicate or not
        is_duplicate = np.any(np.all(self.spots_coords_list == spot_coords))
        if is_duplicate:
            debug_report(f'We already had spot {spot_coords} in cluster{self.cluster_id} :/', debug)
            return

        # Finally! add the new spot to the cluster
        self.add_spot(spot_coords) # no back has been saved yet.
        debug_report(f'finally cluster self.spots_coords_list = {self.spots_coords_list}', debug)

    # C3
    def fill_the_rest(self, debug=False):
        debug_report(f'** running fill_the_rest for cluster{self.cluster_id} with {self.spots_coords_list}', debug)

        self.spots_coords_list = sorted(self.spots_coords_list, key=lambda x: x[0])
        spots_coords_array = np.array(self.spots_coords_list)
        self.avg_x = np.mean(spots_coords_array[:, 0])
        self.avg_y = np.mean(spots_coords_array[:, 1])
        self.avg_r = np.mean(spots_coords_array[:, 2])
        debug_report(f'avg x,y,r ={(self.avg_x, self.avg_y, self.avg_r)}', debug)
        self.mean_fg_list = [None] * len(self.spots_coords_list)
        self.mean_bg_list = [None] * len(self.spots_coords_list)

        self.min_max_abs_coords = CommonFunctions.find_min_max_coords(
            input_coords_list=self.spots_coords_list,
            # min_max_dict=self.min_max_abs_coords
            min_max_dict={'min_x': None, 'max_x': None, 'min_y': None, 'max_y': None}
        )
        debug_report(f'self.min_max_abs_coords = {self.min_max_abs_coords}', debug)

        if self.color:
            return
        self.color = CommonFunctions.give_me_color_for_cluster_id(self.cluster_id)


    # C6
    def init_cluster_based_on_ref_outside_of_block(self, displacement_vec, ref_cluster, debug=False):
        debug_report(
            f'**running init_cluster_based_on_ref_outside_of_block for cluster{self.cluster_id} in block{self.block_id}, based on cluster{ref_cluster.cluster_id} in block{ref_cluster.block_id}',
            debug)
        ref_cluster = deepcopy(ref_cluster)
        debug_report(
            f'ref_cluster -> {ref_cluster.spots_coords_list}\nor IN BLOCK:{ref_cluster.spots_coords_in_block_list}',
            debug)

        actual_displacement_vec = [
            displacement_vec[0] + self.block_start_x - ref_cluster.block_start_x,
            displacement_vec[1] + self.block_start_y - ref_cluster.block_start_y,
            displacement_vec[2]
        ]

        debug_report(f'displacement={displacement_vec} and actual_displacement={actual_displacement_vec}', debug)
        self.spots_coords_list = ref_cluster.spots_coords_list
        for coord in self.spots_coords_list:
            self.move_spots(coord, [actual_displacement_vec], debug)
        debug_report(f'after moving based on actual_displacement_vec -> {self.spots_coords_list}', debug)

        self.fill_the_rest()
        self.add_spots_coords_in_block(debug=debug)
        self.save_coords_backup(debug=debug)
        debug_report(f'new_cluster -> {self.spots_coords_list} or IN BLOCK:{self.spots_coords_in_block_list}', debug)

    def init_cluster_based_on_ref_within_block(self, ref_cluster, debug=False):
        debug_report(
            f'**running init_cluster_based_on_ref_within_block for cluster{self.cluster_id} based on cluster{ref_cluster.cluster_id} in block{ref_cluster.block_id}',
            debug)
        ref_cluster = deepcopy(ref_cluster)
        debug_report(f'ref_cluster: {ref_cluster.full_report(debug)}\nnew_cluster: {self.full_report(debug)}', debug)
        self.spots_coords_list = ref_cluster.spots_coords_list
        self.fill_the_rest()
        self.add_block_info(block_id=ref_cluster.block_id, debug=debug)
        self.add_spots_coords_in_block()
        self.save_coords_backup(debug=debug)
        debug_report(f'in the end --> new_cluster: {self.full_report(debug)}', debug)

    # C1
    def add_block_info(self, block_id, debug=False):
        debug_report(f'** running add_block_related_info for cluster{self.cluster_id} in block{block_id}', debug)
        debug_report(
            f"""before (cluster{self.cluster_id}): 
            block_id={self.block_id}, 
            block_start_x={self.block_start_x}, 
            block_start_y={self.block_start_y}""", debug
        )

        block = ScanDataObj.get_scan_data(file_name=self.file_name).get_block(block_id=block_id)
        self.block_id = block_id
        self.block_start_x = block.start_x
        self.block_start_y = block.start_y
        # self.spots_coords_in_block_list = []
        # self.add_spots_coords_in_block(debug=debug)
        debug_report(
            f"""in the end (cluster{self.cluster_id}): 
                    block_id={self.block_id}, 
                    block_start_x={self.block_start_x}, 
                    block_start_y={self.block_start_y}""", debug
        )

    # C2
    def add_spots_coords_in_block(self, debug=False):
        debug_report(f'** C2: running add_spots_coords_in_block for cluster{self.cluster_id}', debug)
        debug_report(f"before anything: {self.full_report(return_str=debug)}", debug)

        self.spots_coords_in_block_list = []
        for x, y, r in self.spots_coords_list: # loop over all spots in these cluster
            self.spots_coords_in_block_list.append(np.array([int(x-self.block_start_x), int(y-self.block_start_y, r)]))

        self.avg_x_in_block = int(np.mean([arr[0] for arr in self.spots_coords_in_block_list]))
        self.avg_y_in_block = int(np.mean([arr[1] for arr in self.spots_coords_in_block_list]))
        self.add_position_ind(debug=debug)
        debug_report(f"in the end: {self.full_report(debug)}", debug)


    def reverse_add_spots_coords_in_block(self, debug=False):
        self.spots_coords_list = []
        for x, y, r in self.spots_coords_in_block_list:
            new_coords = np.array([x + self.block_start_x, y + self.block_start_y, r])
            self.spots_coords_list.append(new_coords)
        self.avg_x_in_block = np.mean([arr[0] for arr in self.spots_coords_in_block_list])
        self.avg_y_in_block = np.mean([arr[1] for arr in self.spots_coords_in_block_list])

    # C7
    def save_coords_backup(self, debug=False):
        debug_report(f'** saving coords backup for cluster{self.cluster_id}', debug)
        debug_report(
            f'Before: backup_coords_list={self.backup_coords_list}\nbackup_block_coords_list={self.backup_block_coords_list}',
            debug)
        self.backup_coords_list = self.spots_coords_list.copy()
        self.backup_block_coords_list = self.spots_coords_in_block_list.copy()
        debug_report(
            f'After: backup_coords_list={self.backup_coords_list}\nbackup_block_coords_list={self.backup_block_coords_list}\n',
            debug)

    # C4
    def restore_coords_backup_and_more(self, debug=False):
        debug_report(f'** running restore_coords_backup_and_more cluster{self.cluster_id}', debug)
        debug_report(f'before: #spots={len(self.spots_coords_list)}', debug)
        self.spots_coords_list = self.backup_coords_list.copy()
        self.spots_coords_in_block_list = self.backup_block_coords_list.copy()
        self.fill_the_rest()
        self.add_spots_coords_in_block(debug=debug)
        debug_report(f'after: #spots={len(self.spots_coords_list)}\n', debug)

    def add_spots(self, count, direction, debug=False):  # direction: 'r' or 'right'/ 'l' or 'left'
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        avg_cAb_dist = data_obj.avg_spot_distance

        debug_report(
            f'**running add_spots for cluster{self.cluster_id} -> count={count}, direction={direction}, avg_cAb_dist={avg_cAb_dist}',
            debug)

        self.spots_coords_list = sorted(self.spots_coords_list, key=lambda x: x[0])

        if len(self.spots_coords_list) < 2:
            avg_x_dist = avg_cAb_dist
        else:
            avg_x_dist = int(np.mean(np.diff([s[0] for s in self.spots_coords_list])))

        if direction in ["right", "r"]:
            ref_spot_coords = self.spots_coords_list[-1]
        elif direction in ["left", "l"]:
            ref_spot_coords = self.spots_coords_list[0]
            avg_x_dist = -avg_x_dist
        else:
            print(f'{direction} is not supported! use either "right"/"r" or "left"/"l"')
            ref_spot_coords = [0,0,0]
        debug_report(f'reference spot coords={ref_spot_coords} and avg_x_dist={avg_x_dist}', debug)

        for i in range(int(count)):
            new_spot_coords = ref_spot_coords.copy()
            new_spot_coords[0] += avg_x_dist * (i + 1)
            self.spots_coords_list.append(new_spot_coords)
            debug_report(f'added a new spot->{new_spot_coords}', debug)

        self.spots_coords_list = sorted(self.spots_coords_list, key=lambda x: x[0])
        debug_report(f'Final coords: {self.spots_coords_list}', debug)

        self.fill_the_rest()
        self.add_spots_coords_in_block(debug=debug)

    def change_radius_of_spot(self, spot_ind, r_change, debug=False):
        debug_report(
            f'running change_radius_of_spot for cluster{self.cluster_id} with spots_coords_list={self.spots_coords_list}',
            debug)
        coords = self.spots_coords_list[spot_ind].copy()
        coords[2] += int(r_change)
        self.spots_coords_list[spot_ind] = coords.copy()
        debug_report(f'After -> spots_coords_list={self.spots_coords_list}', debug)

    #     def add_or_delete_spots(self, prompt_vec, with_restore=True, debug=False):
    #         debug_report(f'running the "add_or_delete_spots" function for cluster {self.cluster_id} with prompt: {prompt_vec}', debug)

    #         if with_restore:
    #             self.restore_coords_backup_and_more(debug)

    #         self.spots_coords_list = sorted(self.spots_coords_list, key=lambda x: x[0])
    #         debug_report(f'current spots: {self.spots_coords_list}', debug)

    #         prompt_vec_list = read_prompt(prompt_vec, debug)
    #         for count, direction in prompt_vec_list:
    #             if count > 0:
    #                 self.add_spots(count, direction, debug)
    #             else:
    #                 self.delete_relative_spots(count, direction, debug)

    #         self.fill_the_rest()
    #         self.add_spots_coords_in_block()

    # C5
    def manual_edit_spots_in_cluster(self, commands_list=None, debug=False):
        debug_report(
            f'**running the "manual_edit_spots_in_cluster" function for cluster{self.cluster_id} with {commands_list}',
            debug)
        data_obj = ScanDataObj.get_scan_data(self.file_name)
        avg_spot_dist = data_obj.avg_spot_distance

        if not commands_list:
            return
        """ prompt samples: 
            - to delete spot(s): del + spot + position(s) 
                            -> 'del spot-1, 2', 'del spot0,1,2,3'

            - to add spot(s): add + count + left(l)/right(r) 
                            -> 'add 1 to r', 'add 2 to left' 

            - to move spots: move + spot + position/"all" + number_of_pixels + up(u)/down(d)/left(l)/right(r) 
                            -> 'move spot2 10 up, 5 right', 'move all 20 left'

            - to change radius: change_r + spot + position + r + pixel_change_value
                            -> 'change_r spot3 r+2', 'change_r spot6 r-1'



        ** positions are: 0,1,2,3,.... or ...,-3,-2,-1

        """

        self.spots_coords_list = sorted(self.spots_coords_list, key=lambda x: x[0])
        debug_report(f'Starting coords: {self.spots_coords_list}', debug)

        action_dict = {}
        for command in commands_list:
            try:
                action_dict = ClassesFunctions.read_command(command, debug=debug)
            except Exception as e:
                print(f'something wrong with command {command}...')
                print(f"Exception:\n{e}")
                tb = traceback.format_exc()
                print(f"Traceback:\n{tb}")

            if action_dict is None:
                print(f'skipping this one...')
                continue

            action = action_dict['action']
            params = action_dict['params']

            if action == "add_cluster":
                block = data_obj.get_block(self.block_id)
                count = params['count']
                direction = params['direction']
                distance = params['distance']
                distance = avg_spot_dist if distance is None else distance
                max_id = ClassesFunctions.get_max_cluster_id(file_name=self.file_name)
                for i in range(int(count)):
                    displacement_prompt = [(i + 1) * distance, direction]
                    ind = max_id + i + 1
                    new_cluster = Cluster(cluster_id=ind, file_name=self.file_name)
                    new_cluster.init_cluster_based_on_ref_within_block(self, debug)
                    new_cluster.move_some_spots_in_cluster('all', [displacement_prompt], debug)
                    #                     self.add_cluster_id_to_block(ind) ????
                    block.add_cluster_related_info(cluster_id=ind)
                    data_obj.add_new_cluster_to_dict(new_cluster)
                    new_cluster.already_restored = True
                    block.update_min_max_coords_of_clusters(debug=debug)
                    block.update_block_start_end_from_clusters_min_max(debug=debug)
                    block.add_cropped_images(debug=debug)
                    debug_report(f'added cluster {max_id + i + 1} by moving {self.cluster_id} by {avg_spot_dist}', debug)

            elif action == "add":
                if not self.already_restored:
                    self.restore_coords_backup_and_more(debug=debug)
                    self.already_restored = True
                count = params['count']
                direction = params['direction']
                self.add_spots(count, direction, debug=debug)

            elif action == "move":
                if not self.already_restored:
                    self.restore_coords_backup_and_more(debug=debug)
                    self.already_restored = True
                target = params['target']
                movement_prompts = params['movements']
                self.move_some_spots_in_cluster(target, movement_prompts, debug=debug)

            elif action == "change_r":
                if not self.already_restored:
                    self.restore_coords_backup_and_more(debug=debug)
                    self.already_restored = True
                spot = params['spot']
                change = params['change']
                if spot == 'all':
                    for sind in range(len(self.spots_coords_list)):
                        self.change_radius_of_spot(sind, change, debug)
                else:
                    self.change_radius_of_spot(spot, change, debug)

            elif action == "delete":
                if not self.already_restored:
                    self.restore_coords_backup_and_more(debug=debug)
                    self.already_restored = True
                spots = params['spots']
                block = data_obj.get_block(block_id=self.block_id)
                if spots == 'all':
                    block.clusters_ids_list.remove(self.cluster_id)
                    data_obj.delete_cluster(self.cluster_id)
                    debug_report(f'just deleted the whole cluster...', debug)
                    continue

                for spot in spots:
                    a = self.spots_coords_list.pop(spot)
                    debug_report(f'popped {a} from the {self.spots_coords_list} :)', debug)
                    # if spot == -1 and a[0] == block.min_max_coords_of_clusters['max_x']:
                    #     block.min_max_coords_of_clusters['max_x'] -= 10
                    # if spot == 0 and a[0] == block.min_max_coords_of_clusters['min_x']:
                    #     block.min_max_coords_of_clusters['min_x'] += 10

            # TODO: this line runs two times....
            debug_report(f'Coords after {command}: {self.spots_coords_list}', debug)

        self.already_restored = False  # ready for repeats
        self.add_block_info(block_id=self.block_id, debug=debug)
        self.add_spots_coords_in_block(debug=debug)
        self.fill_the_rest(debug=debug)
        self.spots_coords_list = sorted(self.spots_coords_list, key=lambda x: x[0])
        debug_report(f'Final coords: {self.spots_coords_list}', debug)

    # C8
    def move_spots(self, input_coords, coords_displacement_list, debug):
        debug_report(
            f'**running move_spots for cluster{self.cluster_id} on input_coords={input_coords} with coords_displacement_list={coords_displacement_list}',
            debug)

        #         coords = deepcopy(input_coords)
        coords = input_coords
        for displacement in coords_displacement_list:
            debug_report(f'working on displacement {displacement}', debug)

            coords[0] += int(displacement[0])
            coords[1] += int(displacement[1])
            coords[2] += int(displacement[2])

        debug_report(f'output coords={coords}', debug)
        return coords

    #     def edit_cluster_coords(self, prompt_vec, debug=False):
    #         debug_report(f'running the "edit_cluster_coords" function for cluster {self.cluster_id}:', debug)

    #         displacement_list = give_displacement_list_from_prompt_vec(prompt_vec, debug) ????
    #         for displacement in displacement_list:
    #             self.move_spots(displacement,debug)move_some_spots_in_cluster

    def move_some_spots_in_cluster(self, target_spot, displacements_prompt, debug=False):
        debug_report(
            f'**running move_some_spots_in_cluster for cluster{self.cluster_id} for target_spot={target_spot} & displacements_prompt={displacements_prompt}',
            debug)
        debug_report(f'Before:\n\tspots_coords_list={self.spots_coords_list}\n\tspots_coords_in_block_list={self.spots_coords_in_block_list}', debug)

        coords_displacement_list = ClassesFunctions.give_coords_displacement_list_from_prompt_list(displacements_prompt, debug)

        ref_coords = self.spots_coords_list.copy()

        if target_spot == 'all':
            for ind in range(len(ref_coords)):
                changed_coords = self.move_spots(ref_coords[ind], coords_displacement_list, debug)
                ref_coords[ind] = changed_coords.copy()
        else:
            changed_coords = self.move_spots(ref_coords[target_spot], coords_displacement_list, debug)
            ref_coords[target_spot] = changed_coords.copy()

        self.spots_coords_list = ref_coords.copy()
        self.fill_the_rest()
        self.add_spots_coords_in_block(debug=debug)
        debug_report(f'After:\n\tspots_coords_list={self.spots_coords_list}\n\tspots_coords_in_block_list={self.spots_coords_in_block_list}', debug)

    def plot_cluster_spot_measurements(self, bg_label=np.array([]), fg_label=np.array([]), spot_id=None,
                                       padding=10, debug=False, values=None):
        debug_report(f'** running plot_cluster_spot_measurements for cluster{self.cluster_id}', debug)

        if not values:
            values = {'bg': None, 'fg': None}
        if not bg_label.any() or not fg_label.any():
            debug_report(f'** no bg/fg labels, so gonna call measure_all_fg_bg_of_a_cluster....', debug)
            self.measure_all_fg_bg_of_a_cluster(debug=debug)

        block_log_image = ScanDataObj.get_block_image(
            file_name=self.file_name, block_id=self.block_id, image_tag='scaled_image'
        )

        bg_image = np.where((bg_label == 1), block_log_image, np.nan)
        fg_image = np.where((fg_label == 1), block_log_image, np.nan)

        spot_image = np.where((fg_label == 1) | (bg_label == 1), block_log_image, np.nan)

        bg_border = np.argwhere(~np.isnan(bg_image))
        start = np.min(bg_border, axis=0) - padding
        end = np.max(bg_border, axis=0) + padding

        bg_image_cut = CommonFunctions.make_3D_image(bg_image[start[0]:end[0], start[1]:end[1]])
        fg_image_cut = CommonFunctions.make_3D_image(fg_image[start[0]:end[0], start[1]:end[1]])
        spot_image_cut = CommonFunctions.make_3D_image(spot_image[start[0]:end[0], start[1]:end[1]])

        print(
            f'\n\nSpot{spot_id} in cluster{self.cluster_id}: BG median={values["bg"]:.0f}, FG mean={values["fg"]:.0f}',
            end='')
        concat_img1 = CommonFunctions.pad_and_concat_images(bg_image_cut, fg_image_cut, axis=1, pad_value=255)
        concat_img2 = CommonFunctions.pad_and_concat_images(concat_img1, spot_image_cut, axis=1, pad_value=255)
        CommonFunctions.display_in_console(concat_img2, fig_size=[5,5])

    # change the name
    def measure_all_fg_bg_of_a_cluster(self, neg_fg_mask=np.array([]), block_image=np.array([]), debug=False,
                                       plot_images=False, fg_inc_pixels=2, margin_pixels=2, bg_r=4):
        debug_report(f'** running measure_all_fg_bg_of_a_cluster for cluster{self.cluster_id}', debug)
        debug_report(f'neg_fg_mask is {neg_fg_mask.shape} and block_image is {block_image.shape}', debug)
        data_obj = ScanDataObj.get_scan_data(self.file_name)

        if not neg_fg_mask.any():
            block = data_obj.get_block(self.block_id)
            neg_fg_mask = block.make_neg_fg_mask(debug=debug)
            debug_report(f'created the neg_fg_mask {neg_fg_mask.shape}', debug)

        if not block_image.any():
            block_image = ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='image')
            debug_report(f'got the block_image from ScanDataObj {block_image.shape}', debug)

        block_image = deepcopy(block_image)

        i = 0
        debug_report(f'spots_coords_in_block_list -> {self.spots_coords_in_block_list}', debug)

        for x, y, r in self.spots_coords_in_block_list:
            r += fg_inc_pixels

            # background
            spot_bg = np.zeros(block_image.shape)
            cv2.circle(spot_bg, (x, y), round(r * bg_r), 1, thickness=-1)  # creating a big white circle around 1 spot
            spot_bg = np.logical_and(spot_bg, neg_fg_mask)  # remove the fg + margin of all spots in block
            bg_label, _ = ndimage.label(spot_bg)
            bg_mean = ndimage.median(block_image, bg_label)
            #             self.bg_mask_list[i]=bg_label
            self.mean_bg_list[i] = bg_mean

            # foreground
            spot_fg = np.zeros(block_image.shape)
            cv2.circle(spot_fg, (x, y), r, 1, thickness=-1)
            fg_label, _ = ndimage.label(spot_fg)
            fg_mean = ndimage.mean(block_image, fg_label)
            #             self.fg_mask_list[i]=fg_label
            self.mean_fg_list[i] = fg_mean

            if plot_images or debug:
                self.plot_cluster_spot_measurements(
                    bg_label=bg_label,
                    fg_label=fg_label,
                    spot_id=i,
                    debug=debug,
                    values={'bg': bg_mean, 'fg': fg_mean}
                )

            i += 1

        debug_report(f'In the end, mean_bg_list={self.mean_bg_list}, mean_fg_list={self.mean_fg_list}\n', debug)

    def init_cluster_based_on_coords_in_block(self, spots_coords_in_block_list, displacement_list=None, debug=False):
        if not displacement_list:
            displacement_list = []

        debug_report(
            f'**running init_cluster_based_on_coords_in_block with displacement = {displacement_list}, and coords = {spots_coords_in_block_list}')

        debug_report(f'before anything self.spots_coords_in_block_list = {self.spots_coords_in_block_list}')

        self.spots_coords_in_block_list = spots_coords_in_block_list

        debug_report(f'now self.spots_coords_in_block_list = {self.spots_coords_in_block_list}')

        self.spots_coords_list = []
        for x, y, r in self.spots_coords_in_block_list:
            new_coords = np.array([x + self.block_start_x, y + self.block_start_y, r])
            self.spots_coords_list.append(new_coords)
            self.fill_the_rest()
            self.save_coords_backup(debug=debug)

        #         ??????
        for displacement in displacement_list:
            self.move_some_spots_in_cluster(displacement, debug)  # ////////

        debug_report(f'final coords self.spots_coords_in_block_list = {self.spots_coords_list}')

    #     ??????
    def add_position_ind(self, debug=False):
        # data_obj = ScanDataObj.get_scan_data(self.file_name)
        # avg_cAb_dist = data_obj.avg_spot_distance
        # scan_size = data_obj.scan_size
        #         debug_report(f'**running add_position_ind for cluster{self.cluster_id}', debug)

        #         self.position_ind = 0 if self.avg_x_in_block <= avg_block_width/2 else 1 #ONLY 64 blocks chip....
        self.position_ind = 0

    def add_label_on_image_for_cluster(self, image, name_tag, font_scale=1.1, font_thickness=2,
                                       in_block=True, debug=False):
        #         debug=True

        debug_report(f'**running add_label_on_image_for_cluster for cluster{self.cluster_id}', debug)

        text_size = cv2.getTextSize(str(self.cluster_id), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        debug_report(f'text_size: {text_size}', debug)

        coords_list = self.spots_coords_in_block_list if in_block else self.spots_coords_list
        b_coords_list = self.spots_coords_list

        if self.position_ind is None:
            self.add_position_ind(debug=debug)

        if self.position_ind == 0:
            x, y, r = coords_list[0]
            dx = -20 * font_scale * 2 + 10
        else:
            x, y, r = coords_list[-1]
            dx = 20 * font_scale * 2 + 5

        text_x = x - text_size[0] // 2 + dx
        text_y = y + text_size[1] // 2 - 10

        cv2.putText(image, str(name_tag), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, self.color, font_thickness, cv2.LINE_AA)

        if debug:
            cv2.putText(image, f'{(int(coords_list[0][0]), int(coords_list[0][1]))}/{(int(b_coords_list[0][0]), int(b_coords_list[0][1]))}',
                        (text_x - 30, text_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        debug_report(
            f'for cluster{self.cluster_id}: added the text to {(x, y)} since the position_ind={self.position_ind}',
            debug)

    #         debug_report(f'based on avg_x = {self.avg_x_in_block} -> position_ind = {self.position_ind}', debug)
    #         debug_report(f'Function add_position_ind is under construction! -> {self.position_ind}', debug=True)

    def sort_coords_lists(self):
        self.spots_coords_in_block_list = sorted(self.spots_coords_in_block_list, key=lambda x: x[0])
        self.spots_coords_list = sorted(self.spots_coords_list, key=lambda x: x[0])

    def plot_cluster_on_image(self, image, displacement=None, size=2, in_block=True, name_tag=None, debug=False):
        debug_report(f'**running plot_cluster_on_image for cluster{self.cluster_id} with in_block={in_block}', debug)
        if displacement is None:
            displacement = [0, 0]
        self.sort_coords_lists()
        coords_list = self.spots_coords_in_block_list if in_block else self.spots_coords_list
        debug_report(f'sorted coords_list: {coords_list}', debug)

        if not coords_list:
            debug_report(f'!!! For some weird reason, coords_list is empty!!', debug)
            return

        for x, y, r in coords_list:
            cv2.circle(image, (x + displacement[0], y + displacement[0]), r, self.color, size)
        #             debug_report(f'(x,y,r)={(x,y,r)} & displacement={displacement}', debug)

        if not name_tag:
            name_tag = self.cluster_id
        self.add_label_on_image_for_cluster(image=image, in_block=in_block, name_tag=name_tag, debug=debug,
                                            font_scale=int(size / 2), font_thickness=size)

        debug_report(f'plotted cluster{self.cluster_id} with color = {self.color}', debug)


    # C9
    def get_avg_distance_to_another_cluster(self, other_cluster_id, offsets_vec=None,
                                            distance_thr=None, debug=False):
        #         debug=True
        debug_report(
            f'** running get_avg_distance_to_another_cluster for self.cluster{self.cluster_id} with {other_cluster_id}\n',
            debug)

        matched_spots_dict, dont_use_these = self.compare_to_another_cluster_spot_wise(
            other_cluster_id=other_cluster_id,
            offsets_vec=offsets_vec,
            distance_thr=distance_thr,
            debug=debug
        )

        distance_vec_list = []
        for i, j in matched_spots_dict:
            other_coords, self_coords = matched_spots_dict[(i, j)]
            distance = (self_coords - other_coords).astype('int16')
            distance_vec_list.append(distance)
        debug_report(f'distance_vec_list: {distance_vec_list}', debug)
        if distance_vec_list:
            avg_distance = np.mean(distance_vec_list, axis=0).astype('int16')
        else:
            avg_distance = np.array([0, 0, 0])
        debug_report(f'new avg_distance: {avg_distance}', debug)
        return avg_distance

    # def optimize_cluster_coords(self, size=500, debug=False, plot_images=False):
    #     # first time this function is called in 'final_edits_after_adding_clusters_to_block' in 'connect_clusters_to_blocks'
    #     debug_report(f'** running optimize_cluster_coords for cluster{self.cluster_id}', debug)
    #
    #     # this shouldn't happen
    #     if self.block_id is None:
    #         print(f"cluster{self.cluster_id} has not block assigned to it. So I'm gonna skip!")
    #         return
    #
    #     data_obj = ScanDataObj.get_scan_data(self.file_name)
    #     block = data_obj.get_block(self.block_id)
    #
    #     # todo
    #     if debug or plot_images:
    #         print('\n\nThis is before anything:\n\n', self.__dict__)
    #         block.plot_block(pic_size=500)
    #
    #     block_image = ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id, image_tag='image')
    #
    #     new_coords_list, highest_intensities_per_spot = CommonFunctions.optimized_spots_coords(
    #         input_image=block_image,
    #         coords_list=self.spots_coords_in_block_list,
    #         debug=debug
    #     )
    #
    #     self.update_cluster_coords_list(new_coords_list, debug=debug)
    #     debug_report(f"""new coords in block: {self.spots_coords_in_block_list}, absolute coords: {self.spots_coords_list}""", debug)
    #     self.already_restored = True # check! i dont like this...
    #     self.save_coords_backup(debug=debug)
    #     if plot_images:
    #         block.plot_block(pic_size=500, debug=1)
    #         block_scaled_image = ScanDataObj.get_block_image(file_name=self.file_name, block_id=self.block_id,
    #                                                   image_tag='scaled_image')
    #         colored_image = CommonFunctions.make_3D_image(block_scaled_image)
    #         mean_cluster_signal = int(np.mean(highest_intensities_per_spot))
    #         self.plot_cluster_on_image(colored_image, name_tag=mean_cluster_signal, debug=debug)
    #         CommonFunctions.display_in_console(colored_image, fig_size=size)
    #
    #     if debug or plot_images:
    #         debug_report(f'\n\nThis is in the end: {self.full_report(debug)}', debug)
    #         block.plot_block(pic_size=500, debug=1)

    # check: i dont like this! i dont like the reverse part...
    def update_cluster_coords_list(self, new_coords_list, debug=False):
        #         debug=True
        debug_report(f'old_coords_list: {self.spots_coords_in_block_list}, new_coords_list: {new_coords_list}\n', debug)
        self.spots_coords_in_block_list = deepcopy(new_coords_list)
        self.reverse_add_spots_coords_in_block(debug=debug)
        self.fill_the_rest()
        self.spots_coords_in_block_list = sorted(self.spots_coords_in_block_list, key=lambda x: x[0])
        self.spots_coords_list = sorted(self.spots_coords_list, key=lambda x: x[0])
        debug_report(self.full_report(debug), debug)


    # C11
    def compare_to_another_cluster_spot_wise(self, other_cluster_id, offsets_vec=None, distance_thr=None, debug=False):
        debug_report(f'** running "compare_to_another_cluster_spot_wise" for self.cluster{self.cluster_id} with other_cluster{other_cluster_id}', debug)

        data_obj = ScanDataObj.get_scan_data(self.file_name)
        scan_size = data_obj.scan_size

        if not distance_thr:
            distance_thr = [100/scan_size, 50/scan_size]
        if offsets_vec is None:
            offsets_vec = [0, 0, 0]
        if len(offsets_vec) == 2:
            offsets_vec.append(0) # for radius

        other_cluster = data_obj.get_cluster(other_cluster_id)
        debug_report(
            f'self cluster is: {self.full_report(return_str=debug)}\nother cluster is {other_cluster.full_report(return_str=debug)}', debug)
        debug_report(f'offset={offsets_vec}\ndistance_thr={distance_thr}', debug)

        other_c_coords_list = [coord + np.array(offsets_vec) for coord in other_cluster.spots_coords_in_block_list]
        debug_report(f'with offset of {offsets_vec}, changed the other_coords to: {other_c_coords_list}', debug)

        self_c_coords_list = deepcopy(self.spots_coords_in_block_list)
        dont_use_these = {'for_other': [], 'for_self': []}
        matched_spots_dict = {}

        for i, other_coords in enumerate(other_c_coords_list):
            if i in dont_use_these['for_other']:
                continue
            for j, self_coords in enumerate(self_c_coords_list):
                if j in dont_use_these['for_self']:
                    continue
                dx, dy, dr = np.abs(other_coords - self_coords)
                if dx > distance_thr[0] or dy > distance_thr[1]:  # two different spots
                    continue
                matched_spots_dict[(i, j)] = (other_coords, self_coords)
                debug_report(f'found a match! -> other:{other_coords} (i={i}) & self:{self_coords} (j={j})', debug)
                dont_use_these['for_other'].append(i)
                dont_use_these['for_self'].append(j)
        debug_report(f'dont_use_these: {dont_use_these}', debug)
        return matched_spots_dict, dont_use_these


    # C10
    def merge_with_another_cluster(self, other_cluster_id, offsets_vec=None,
                                   distance_thr=None, debug=False):
        #         debug=True
        debug_report(
            f'** running merge_with_another_cluster for self.cluster{self.cluster_id} with {other_cluster_id}\n',
            debug)

        matched_spots_dict, dont_use_these = self.compare_to_another_cluster_spot_wise(
            other_cluster_id=other_cluster_id,
            offsets_vec=offsets_vec,
            distance_thr=distance_thr,
            debug=debug
        )

        new_coords_list = []
        for i, j in matched_spots_dict:
            other_coords, self_coords = matched_spots_dict[(i,j)]
            # todo
            new_coords = np.round(np.mean(np.array([other_coords,self_coords, self_coords]), axis=0)).astype('int16')
            # new_coords = np.array(self_coords) + np.array(offsets_vec)
            # new_coords = np.array(self_coords)
            #                 new_coords = np.round(np.mean(np.array([temp_other,temp_self,temp_self]), axis=0)).astype('int16')
            debug_report(f'for the match between other:{other_coords} & self:{self_coords} => new_coords={new_coords}', debug)
            new_coords_list.append(new_coords)

        # now adding the spots that didn't have any matches :)
        other_cluster = ScanDataObj.get_scan_data(self.file_name).get_cluster(other_cluster_id)
        for i, other_coords in enumerate(other_cluster.spots_coords_in_block_list):
            if i not in dont_use_these['for_other']:
                new_coords_list.append(np.array(offsets_vec)+np.array(other_coords))
                # new_coords_list.append(other_coords)

        for j, self_coords in enumerate(self.spots_coords_in_block_list):
            if j not in dont_use_these['for_self']:
                new_coords_list.append(self_coords.copy())

        debug_report(f'final new_coords_list: {new_coords_list}', debug)
        self.update_cluster_coords_list(new_coords_list=new_coords_list, debug=debug)
        self.already_restored = True



    def full_report(self, return_str=None):
        if return_str in [False, 0]:
            return
        skip_attrs = ['file_name','mean_fg_list','mean_bg_list','signal_list','clean_signal_list',
                     'short_clean_signal_list','mean_cluster_signal','color','position_ind','name']
        coord_attrs = ['spots_coords_list','spots_coords_in_block_list',
                       'backup_coords_list','backup_block_coords_list']
        output = '\n\n'
        for attr, value in self.__dict__.items():
            if attr in skip_attrs:
                continue
            if attr in coord_attrs:
                if value is None or value == []:
                    output += f'{attr}: {value}\n'
                else:
                    output += f'\t{attr}: total spots={len(value)}, spot0={value[0] }\n'
                # output += f'Complete list of {attr}: {value}\n'
            else:
                output += f'\t{attr}: {value}'
        if return_str in [True, 1]:
            return output
        else:
            print(return_str)
