import pickle
import time
from importlib import reload
import re
import traceback
import numpy as np
import pandas as pd
import cv2
from copy import deepcopy
from Classes import BlockObj, ClusterObj, ScanDataObj
from Functions import CommonFunctions
reload(CommonFunctions)
reload(ClusterObj)
reload(BlockObj)
reload(ScanDataObj)
from Functions.CommonFunctions import debug_report


#%% First, init Blocks:
def create_new_block(file_name, block_id, init_offset=None, block_distance_adjustment=None, debug=False):
    debug_report(f'** running "create_new_block" for block{block_id}', debug)
    str_row_number, str_col_number = re.findall(r'\d+', block_id)
    block = BlockObj.Block(
        block_id=block_id,
        file_name=file_name,
        col_number=int(str_col_number),
        row_number=int(str_row_number)
    )
    block.set_start_and_end_of_block(init_offset=init_offset,  debug=debug,
                                     block_distance_adjustment=block_distance_adjustment,)
    block.add_cropped_images(debug=debug)
    block.save_backup(debug=debug) # block level backup
    debug_report(f'created the new block {block.block_id}', debug=debug)
    return block


def init_blocks_dict(file_name, init_offset=None, block_size_adjustment=0, debug_block_ids=None,
                     block_distance_adjustment=None, debug=False, plot_blocks=False):
    debug_report(f'** running "init_blocks_dict"', debug)
    data_obj = ScanDataObj.get_scan_data(file_name)

    if debug_block_ids is None:
        debug_block_ids = []

    data_obj.reset_blocks_dict()
    data_obj.block_size += int(block_size_adjustment)
    if block_distance_adjustment is None:
        block_distance_adjustment = [0,0]
    block_distance_adjustment = [x - block_size_adjustment for x in block_distance_adjustment]
    for r_number in range(data_obj.block_nrow):
        for c_number in range(data_obj.block_ncol):
            block_id = f"r{r_number}c{c_number}"
            debug_block = True if debug or block_id in debug_block_ids else False
            block = create_new_block(file_name=file_name, block_id=block_id, debug=debug_block,
                                     init_offset=init_offset, block_distance_adjustment=block_distance_adjustment)
            if plot_blocks and debug_block:
                block.plot_block(debug=debug_block, fig_size=[5,5])
            data_obj.add_new_block_to_dict(block)
    ScanDataObj.update_scan_data_dict(data_obj)
    debug_report(f'created {len(data_obj.blocks_dict.keys())} blocks', debug)
    if plot_blocks:
        plot_blocks_on_image(file_name=file_name, debug=debug)


#%% Init Clusters
# todo: too many indents!
def create_new_cluster(file_name, cluster_id, debug=False):
    return ClusterObj.Cluster(cluster_id=cluster_id, file_name=file_name)

def init_clusters_dict(circles_coords, predicted_clusters_ids, file_name,
                       debug_clusters_ids=None, debug=False, optimize_spots_coords=True):
    debug_report(f'** running "init_clusters_dict"', debug)
    if debug_clusters_ids is None:
        debug_clusters_ids = []

    sorted_indices = np.argsort(predicted_clusters_ids)
    sorted_clusters_ids = predicted_clusters_ids[sorted_indices]
    sorted_circles_coords = [circles_coords[i] for i in sorted_indices]

    data_obj = ScanDataObj.get_scan_data(file_name)
    data_obj.reset_clusters_dict()
    count = -1
    last_cluster_id = len(sorted_clusters_ids) - 1
    new_cluster = True
    skip_custer = None
    print(f'total detected spots: {len(sorted_clusters_ids)}')
    print(f'Optimized and added spot number ', end='')
    for i, cid in enumerate(sorted_clusters_ids):
        cluster_id = int(cid)
        count+=1

        if cluster_id == -1: # this circle does not belong to any clusters
            continue

        if cluster_id == skip_custer:
            continue
        debug_cluster = debug or cluster_id in debug_clusters_ids
        spot_coords = [int(x) for x in sorted_circles_coords[i]]
        debug_report(f'Looking at spot {i} ({spot_coords}) for cluster{cluster_id}....', debug_cluster)

        if i != last_cluster_id:
            next_cluster_id = int(sorted_clusters_ids[i + 1])
        else:
            next_cluster_id = 'finished!'

        if new_cluster:
            debug_report(f'starting a new cluster with cluster_id={cluster_id}....', debug_cluster)
            data_obj.add_new_cluster_to_dict(create_new_cluster(file_name=file_name, cluster_id=cluster_id, debug=debug_cluster))
            cluster = data_obj.get_cluster(cluster_id)
            block = BlockObj.Block(block_id=None, file_name=file_name)
            for block_id, test_block in data_obj.get_blocks_dict().items():
                debug_report(f'block{block_id}: ({test_block.start_x}-{test_block.end_x}) & ({test_block.start_y}-{test_block.end_y})',
                             debug_cluster)
                if not (test_block.start_x < spot_coords[0] < test_block.end_x and test_block.start_y < spot_coords[1] < test_block.end_y):
                    continue
                debug_report(f'hehe!! spot{i} in {cluster_id} is in {test_block.block_id}', debug_cluster)
                cluster.add_block_info(block_id=block_id, debug=debug_cluster)
                block = test_block
                break
        else:
            debug_report(f'adding a new spot{i} to cluster{cluster_id}....', debug_cluster)
            cluster = data_obj.get_cluster(cluster_id)
            block = data_obj.get_block(cluster.block_id)

        if block.block_id is None:
            # print(f'\ncluster{cluster_id} ({spot_coords}) is not in any of the blocks!',end=' ')
            skip_custer = cluster_id
            continue
        spot_coords_in_block = [spot_coords[0] - block.start_x, spot_coords[1] - block.start_y, spot_coords[2]]
        debug_report(f'spot_coords_in_block={spot_coords_in_block}', debug_cluster)

        block_image = ScanDataObj.get_block_image(file_name=file_name, block_id=block.block_id, image_tag='image')

        if optimize_spots_coords:
            new_coords_list_but_in_block, highest_intensities_per_spot = CommonFunctions.optimized_spots_coords(
                input_image=block_image,
                coords_list=[spot_coords_in_block],
                avg_r=data_obj.avg_spot_r,
                debug=debug_cluster
            )
        else:
            new_coords_list_but_in_block = [spot_coords_in_block]
        new_abs_coords_list = [new_coords_list_but_in_block[0][0] + block.start_x,
                              new_coords_list_but_in_block[0][1] + block.start_y,
                              new_coords_list_but_in_block[0][2]]
        cluster.add_new_spot_to_cluster(spot_coords=new_abs_coords_list, debug=debug_cluster)  # no backup is saved
        block.add_cluster_related_info(cluster_id=cluster_id)
        # cluster.add_block_related_info(block_id=block_id, debug=debug)
        data_obj.add_new_block_to_dict(block=block)
        data_obj.add_new_cluster_to_dict(cluster=cluster)

        if count % 50 == 0:
            print(f'{count}, ', end='')

        if next_cluster_id != cluster_id:
            new_cluster = True
            cluster.add_spots_coords_in_block(debug=debug_cluster)
            cluster.fill_the_rest(debug=debug_cluster)
            cluster.save_coords_backup(debug=debug_cluster)
            data_obj.add_new_cluster_to_dict(cluster=cluster)
            data_obj.save_clusters_dict_backup()
            data_obj.save_blocks_dict_backup()
            ScanDataObj.update_scan_data_dict(data_obj)
        else:
            new_cluster = False
    print('\nDone!')
    return

#%%
def final_edits_after_adding_clusters_to_block(file_name, block_id, debug=False,plot_images=False):
    debug_report(f'** running final_edits_after_adding_clusters_to_block for block{block_id}', debug)

    data_obj = ScanDataObj.get_scan_data(file_name)
    block = data_obj.get_block(block_id=block_id)

    if debug:
        print('\nAt first: ',block.__dict__)

    if not block.clusters_ids_list:
        for cluster in data_obj.get_clusters_dict().values():
            if cluster.block_id == block_id:
                block.add_cluster_related_info(cluster_id=cluster.cluster_id)
    block.cAb_names = data_obj.cAb_names
    block.update_min_max_coords_of_clusters(debug=debug) # this is the first time min_max values are defined
    # block.update_block_start_end_from_clusters_min_max(debug=debug) # first time this is called & no backup is saved
    block.add_cropped_images(debug=debug, plot_images=plot_images)
    block.save_backup(debug=debug)
    data_obj.add_new_block_to_dict(block=block)
    # block.add_cropped_images(debug=debug)
    # if debug:
    #     block.plot_block(pic_size=400, debug=debug)

    debug_report(f'these are the clusters: {block.clusters_ids_list}', debug)
    counter = 0
    if len(data_obj.cAb_names) < len(block.clusters_ids_list):
        print(f'There are {len(block.clusters_ids_list)} clusters in block{block.block_id}, and only {len(data_obj.cAb_names)} cAb names. Will reset to c1,c2,c3,... for now...')
        data_obj.cAb_names = [f'c{i}' for i in range(len(block.clusters_ids_list))]
    for cluster_id in block.clusters_ids_list:
        cluster = data_obj.get_cluster(cluster_id)
        if debug:
            print(cluster.__dict__)
        if cluster is None:
            continue

        cluster.name = data_obj.cAb_names[counter]
        cluster.add_block_info(block_id=block_id, debug=debug)
        cluster.add_spots_coords_in_block(debug=debug)
        # cluster.optimize_cluster_coords(debug=debug, plot_images=debug)  # !????
        cluster.save_coords_backup(debug=debug)
        data_obj.add_new_cluster_to_dict(cluster=cluster)
        counter+=1

    block.update_backup_clusters_ids_list()  # just saves the IDs????
    data_obj.add_new_block_to_dict(block=block)
    data_obj.save_blocks_dict_backup()
    data_obj.save_clusters_dict_backup()
    data_obj.save_blocks_dict_backup()
    ScanDataObj.update_scan_data_dict(data_obj)
    if debug:
        print('\nEnding with:', block.__dict__)

# todo: too many indents
def connect_clusters_to_blocks(file_name, debug=False, plot_images=False, debug_clusters=None, debug_blocks=None):
    debug_report(f'** running "connect_clusters_to_blocks"', debug)

    skip_these = []
    try:
        for block_id in ScanDataObj.get_scan_data(file_name).blocks_dict.keys():  # for all blocks
            start_time = time.time()
            block_debug = debug or block_id in debug_blocks
            final_edits_after_adding_clusters_to_block(
                file_name=file_name,
                block_id=block_id,
                debug=block_debug,
                plot_images=plot_images
            )
            if block_debug:
                print('\nthis is outside of the function', ScanDataObj.get_scan_data(file_name).get_block(block_id=block_id).__dict__)
            process_time = (time.time() - start_time)
            debug_report(f'Done with block{block_id} (checked {len(skip_these)} in {process_time:.2f} seconds)', debug=debug)
        if plot_images:
            plot_blocks_on_image(file_name=file_name,debug=debug)

        data_obj = ScanDataObj.get_scan_data(file_name)
        data_obj.save_clusters_dict_backup()
        data_obj.save_blocks_dict_backup()
        ScanDataObj.update_scan_data_dict(data_obj)
        if debug_blocks:
            print(debug_blocks)
            print('\nand this is the end....', ScanDataObj.get_scan_data(file_name).get_block(block_id=block_id).__dict__)

    except Exception as e:
        print(f"Exception:\n{e}")
        tb = traceback.format_exc()
        print(f"Traceback:\n{tb}")


#%%

def load_block_related_images(file_name, debug=False, plot_images=False):
    data_obj = ScanDataObj.get_scan_data(file_name)
    block_ids_list = data_obj.blocks_dict.keys()
    for block_id in block_ids_list:
        block = data_obj.get_block(block_id=block_id)
        block.add_cropped_images(debug=debug, plot_images=debug)


def read_scan_data_from_pickle(file_name, path, start_over=False, plot_results=False, debug=False):
    # this is when there is no pickle to read from, or we need to reset it
    if start_over:
        print("start_over is True")
        return None

    # if we want to read the scan data from pickles already scanned data (start_over=False)
    try:
        load_current_data_obj(file_name=file_name, path=path)
        loaded_scan_data = ScanDataObj.get_scan_data(file_name=file_name)
        load_block_related_images(file_name=file_name, debug=debug, plot_images=plot_results)

    except Exception:
        print(f"No file with name {file_name}!")
        loaded_scan_data = read_scan_data_from_pickle(file_name=file_name, path=path, start_over=True,
                                                      plot_results=plot_results, debug=debug)
    return loaded_scan_data

#%%
# def create_neg_fg_mask(file_name, spots_coords_list, fg_inc_pixels=1, margin_pixels=3, debug=False):
#     debug_report(f'** running create_neg_fg_mask for all {len(spots_coords_list)} clusters_ids', debug)
#
#     image = ScanDataObj.get_image_from_dict(file_name=file_name, dict_key='file_image')
#     neg_fg_mask = np.ones(image.shape)
#     for x,y,r in spots_coords_list:
#         r += fg_inc_pixels
#         cv2.circle(neg_fg_mask, (x, y), r + margin_pixels, 0, thickness=-1)
#
#     ScanDataObj.add_to_images_dict(file_name=file_name, dict_key='file_neg_fg_mask', dict_value=neg_fg_mask)
#     CommonFunctions.show(255*neg_fg_mask, plot_images=debug)
#     # return neg_fg_mask


#%%
def plot_blocks_on_image(file_name, blocks_ids_lists=None, debug=False, display_in_console=True, text_color=(255, 255, 255)):
    data_obj = ScanDataObj.get_scan_data(file_name)
    blocks_ids_lists = list(data_obj.get_blocks_dict().keys())
    input_image = deepcopy(ScanDataObj.get_image_from_dict(file_name=file_name, dict_key='file_image'))
    h, w = input_image.shape

    if data_obj.assay == 'OF':
        borders_image = deepcopy(input_image)[:h//2,:]
        block_ids_for_plot = ['r0c0', 'r0c2', 'r1c1', 'r2c0', 'r2c2',]

    else:
        borders_image = deepcopy(input_image)[:h//3,:]
        block_ids_for_plot = ['r0c0', 'r0c3', 'r2c0', 'r2c3', 'r4c0', 'r4c3']
    if blocks_ids_lists:
        block_ids_for_plot = blocks_ids_lists
    for block_id in block_ids_for_plot:
        block = data_obj.get_block(block_id)
        cv2.rectangle(borders_image, (block.start_x, block.start_y), (block.end_x, block.end_y), (0, 0, 0), 10)
        cv2.putText(borders_image, block_id, (block.start_x + 10, block.start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 10,
                    text_color, 20)
    if display_in_console:
        CommonFunctions.display_in_console(borders_image)
    return borders_image



#%%
def read_command(command, debug=False):
    #     debug=True
    debug_report(f'**running read_command for {command}', debug)

    # Delete the entire cluster: 'delete' or 'del'
    if command in ['delete', 'del']:
        debug_report(f"Delete Command - Spots: all", debug)
        return {'action': "delete", "params": {"spots": "all"}}

    # Delete spots command: del + spot + position(s) -> 'del spot-1, -2', 'del spot0,1,2,3'
    elif command.startswith('del'):
        match = re.search(r'del spot((?:-?\d+,?\s*)+)', command)
        spots_str = match.group(1)
        spots = [int(spot) for spot in re.split(r',\s*', spots_str.strip())]
        debug_report(f"Delete Command - Spots: {spots}", debug)
        return {'action': "delete", "params": {"spots": spots}}

    # Add an entire cluster, based on this one: add + count + above/u/up) or below/d/down -> 'add 1 cluster above', 'add 2 clusters below'
    elif 'cluster' in command:
        pattern = r'add (\d+) cluster(?:s)? (above|below|u|d|up|down)(?:.*?d=(\d+))?'
        count, direction, distance = re.search(pattern, command).groups()
        debug_report(f"Add Cluster Command - Count: {int(count)}, Direction: {direction}", debug)
        return {'action': "add_cluster", "params": {"count": int(count), "direction": direction,
                                                    "distance": int(distance) if distance is not None else None}}

    # Add spots command: add + count + left(l)/right(r) -> 'add 1 to r', 'add 2 to left'
    elif command.startswith('add'):
        count, direction = re.search(r'add (\d+) to (r|l|right|left|abs)', command).groups()
        debug_report(f"Add Command - Count: {int(count)}, Direction: {direction}", debug)
        return {'action': "add", "params": {"count": count, "direction": direction}}

    # Move spots command: move + spot + position/"all" + number_of_pixels + up(u)/down(d)/left(l)/right(r) -> 'move spot2 10 up, 5 right', 'move all 20 left'
    elif command.startswith('move'):
        pattern = r'move\s+(spot(-?\d+)|all)\s+((?:\d+\s+(?:up|down|left|right|u|d|l|r)\s*,?\s*)+)'
        match = re.search(pattern, command)
        target = match.group(2) if match.group(2) else match.group(1)
        target = int(target) if target and target.isdigit() or (
                    target and target.startswith('-') and target[1:].isdigit()) else target

        movements_str = match.group(3)
        movements = [m.strip().split() for m in re.split(r',\s*', movements_str)]

        direction_map = {'u': 'up', 'd': 'down', 'l': 'left', 'r': 'right'}
        movements = [[int(m[0]), direction_map.get(m[1], m[1])] for m in movements]

        debug_report(f"Move Command - Target: {target}, Movements: {movements}", debug)
        return {'action': "move", "params": {"target": target, "movements": movements}}

    # Change radius command: change_r + spot + position + r + pixel_change_value -> 'change_r spot3 r+2', 'change_r spot6 r-1'
    elif command.startswith('change_r'):
        match = re.search(r'change_r\s+(spot(-?\d+)|all)\s+r([+-]\d+)', command).groups()

        spot_or_all, spot_number, change = match
        change = int(change)
        if spot_or_all == "all":
            spot = "all"
        else:
            spot = int(spot_number)
        debug_report(f"Change Radius Command - Spot: {spot}, Change: {change}", debug)
        return {'action': "change_r", "params": {"spot": spot, "change": change}}

    elif command.startswith('merge'):
        cluster_id = int(re.search(r'\d+', command).group())
        return {'action': "merge", "params": {"other_cluster_id": cluster_id}}
    else:
        print("Unknown command!!", command)
        return None



def give_coords_displacement_list_from_prompt_list(prompt_list, debug=False):
    debug_report(f'**running give_coords_displacement_list_from_prompt_list with {prompt_list}', debug)
    coords_displacement = []
    for pixel_count, direction in prompt_list:
        if direction in ['up', 'u', 'above']:
            coords_displacement.append([0, -pixel_count, 0])
        elif direction in ['down', 'd', 'below', 'under']:
            coords_displacement.append([0, pixel_count, 0])
        elif direction in ['right', 'r']:
            coords_displacement.append([pixel_count, 0, 0])
        elif direction in ['left', 'l']:
            coords_displacement.append([-pixel_count, 0, 0])
        else:
            print('Please use one of these keywords for direction: [right/r, left/l, up/u/above, down/d/below/under]')
    return coords_displacement


def get_max_cluster_id(file_name):
    return max(ScanDataObj.get_scan_data(file_name=file_name).get_clusters_dict().keys())


def restore_cluster_from_backup(file_name, cluster_id, debug=False):
    debug_report(f'Restoring cluster {cluster_id} from backup_clusters_dict', debug)
    data_obj = ScanDataObj.get_scan_data(file_name=file_name)
    restored_cluster = data_obj.get_cluster_backup(cluster_id=cluster_id)
    debug_report(f'this is the restored version: {restored_cluster.full_report(return_str=debug)}', debug)
    data_obj.add_new_cluster_to_dict(restored_cluster)


def save_current_data_obj(file_name, path=''):
    data_obj = ScanDataObj.get_scan_data(file_name)
    with open(path + f'/{file_name}_data_obj.pickle', 'wb') as file:
        pickle.dump(data_obj, file)

def load_current_data_obj(file_name, path=''):
    data_obj = ScanDataObj.create_new_scan_data(file_name=file_name)
    with open(path + f'/{file_name}_data_obj.pickle', 'rb') as file:
        data_obj = pickle.load(file)
    ScanDataObj.update_scan_data_dict(data_obj)
    return



#check me
def init_from_pickle(path='', debug=False):
    ...
#     global img, scaled_img, clusters_dict, blocks_dict, backup_clusters_dict, backup_blocks_dict, avg_cAb_dist, scan_size, avg_block_width, global_avg_r, images_dict, block_width, block_height, block_ncol, block_nrow, avg_cAb_dist, cAb_names
#     try:
#         CommonFunctions.init()
#
#         image, scaled_logged_image = CommonFunctions.load_image(path + '.tif', debug=False)
#         img = deepcopy(image)
#         scaled_img = deepcopy(scaled_logged_image)
#         print(f'image is {img.shape} and scaled_logged_image is {scaled_img.shape}')
#
#         with open(path + '_parameters.pickle', 'rb') as file:
#             all_params = pickle.load(file)
#
#         avg_cAb_dist = all_params['classes_params']['avg_cAb_dist']
#         avg_block_width = all_params['classes_params']['avg_block_width']
#         scan_size = all_params['classes_params']['scan_size']
#         global_avg_r = all_params['classes_params']['global_avg_r']
#         block_width = all_params['classes_params']['block_width']
#         block_height = all_params['classes_params']['block_height']
#         block_ncol = all_params['classes_params']['block_ncol']
#         block_nrow = all_params['classes_params']['block_nrow']
#         cAb_names = all_params['cAb_names']
#
#         with open(path + '_dicts.pickle', 'rb') as file:
#             clusters_d, blocks_d, backup_clusters_d, backup_blocks_d = pickle.load(file)
#         #             clusters_d, blocks_d = pickle.load(file)
#         print(f'clusters_dict has {len(clusters_d.keys())} clusters')
#         print(f'blocks_dict has {len(blocks_d.keys())} blocks')
#
#         clusters_dict = deepcopy(clusters_d)
#         blocks_dict = deepcopy(blocks_d)
#         backup_clusters_dict = deepcopy(backup_clusters_d)
#         backup_blocks_dict = deepcopy(backup_blocks_d)
#
#         images_dict = {}
#         init_images_dict(input_image=img, input_log_image=scaled_img) #Checkme
#
#         output = {
#             'clusters_dict': clusters_dict,
#             'blocks_dict': blocks_dict,
#             'image': image,
#             'original_log_image': scaled_img,
#             'all_params': all_params
#         }
#     except Exception as e:
#         print(f"Exception:\n{e}\n")
#         tb = traceback.format_exc()
#         print(f"Traceback:\n{tb}\n")
#         output = {}
#     return output
#

# %%


# %%
def give_block_ids_from_r_c(row_vec=range(16), col_vec=[0, 1, 2, 3], debug=False):
    block_ids_list = []
    for r in row_vec:
        for c in col_vec:
            block_id = f"r{r}c{c}"
            block_ids_list.append(block_id)
    return block_ids_list


def edit_multiple_blocks(block_ids_list, file_name, manual_spot_edit_dict=None, init_template_id='r0c0',
                         debug=False, plot_before_after=False, plot_mask=False, plot_final_results=True,
                         move_whole_block_match=None, preprocess_params=None, overwrite=True, debug_blocks=None,
                         redo_circle_finding_for_blocks_or_clusters=None, restore_block_coords=None,
                         correct_N=None, debug_clusters=None, fig_size=None, crop_to_mask=True):
    debug_report(f'** Running edit_multiple_blocks funtion, and init_template_id={init_template_id}', debug)
    # crop_to_mask=False
    if not block_ids_list:
        return
    if debug_blocks is None:
        debug_blocks = []
    if manual_spot_edit_dict is None:
        manual_spot_edit_dict = {}
    if move_whole_block_match is None:
        move_whole_block_match = {}
    if debug_clusters is None:
        debug_clusters = []
    if not redo_circle_finding_for_blocks_or_clusters:
        redo_circle_finding_for_blocks_or_clusters = []
    if not restore_block_coords:
        restore_block_coords = []

    data_obj = ScanDataObj.get_scan_data(file_name=file_name)

    picture = {}
    for block_id in block_ids_list:
        debug_block = True if block_id in debug_blocks else debug
        block = data_obj.get_block(block_id)
        debug_report(f'in the beginning: {block.full_report(return_str=debug_block)}', debug_block)

        if overwrite:
            block.dont_touch_this_block = False

        if block_id in restore_block_coords:
            block.reset_block_start_end_coords(debug=debug_block)
        if block_id in redo_circle_finding_for_blocks_or_clusters or any(c_id in redo_circle_finding_for_blocks_or_clusters for c_id in block.clusters_ids_list) :
            block.redo_circle_finding(target_list=redo_circle_finding_for_blocks_or_clusters, debug=debug_block)

        if block_id == init_template_id or block.dont_touch_this_block:
            debug_report(f"will not be editing block{block_id}...", debug_block)
            pic = block.plot_block(plot_images=False, label=block_id, debug=debug_block,
                                   with_border=False, fig_size=[5, 5], crop_to_mask=crop_to_mask)
            picture[block_id] = pic
            debug_report(f"added its pic {pic.shape} pictures dict:", debug_block)
            if debug:
                CommonFunctions.display_in_console(pic)
            continue

        else:
            block.restore_backup(debug=False, restore_original_clusters_too=True)

    for block_id in block_ids_list:
        debug_block = True if block_id in debug_blocks else debug
        block = data_obj.get_block(block_id)
        if block.dont_touch_this_block:
            continue

        template_id = f"r{block.row_number - 1}c{block.col_number}"
        if template_id not in data_obj.get_blocks_dict().keys():
            template_id = init_template_id
        template_block = data_obj.get_block(template_id)
        debug_report(f'In the loop for block{block_id} and init_template_id is {init_template_id}', debug_block)

        if block_id == init_template_id:
            continue

        if block_id in move_whole_block_match:
            move_match = move_whole_block_match[block_id]
        else:
            move_match = [0, 0]
        debug_report(f'move_match{move_match}', debug=debug_block)

        block.create_clusters_from_another_block(
            template_block_id=template_id,
            plot_images=debug,
            debug=debug_block,
            debug_clusters=debug_clusters,
            move_match=move_match,
            preprocess_params=preprocess_params,
        )

        block.edit_block(manual_spot_edit_dict=manual_spot_edit_dict, with_restore=False,
                         plot_before_after=plot_before_after, debug=debug_block, debug_clusters=debug_clusters)
        #             block.reset_min_max_coords_of_clusters(debug=debug)
        # block.center_block_image(debug=debug_block)
        if not correct_N:
            correct_N = len(template_block.clusters_ids_list)

        # todo: sort based on number of spots!
        if len(block.clusters_ids_list) != correct_N:
            print(
                f'block{block_id} has {len(block.clusters_ids_list)} clusters which is wrong!\n->{block.clusters_ids_list}')
            block.plot_block(debug=debug_block)

        block_mask = block.create_block_mask(debug=debug_block, plot_images=plot_mask)
        if plot_final_results:
            picture[block_id] = block.plot_block(plot_images=False, label=block_id, with_border=False,
                                                 fig_size=300, crop_to_mask=crop_to_mask, debug=debug_block)

        debug_report(f'at the end: {block.full_report(return_str=debug)}', debug_block)
    if plot_final_results:
        do_final_results_plot(file_name=file_name, block_ids_list=block_ids_list, picture=picture, debug=debug,
                              fig_size=fig_size, crop_to_mask=crop_to_mask)
#         else:
#             for pic in picture:
#                 CommonFunctions.display_in_console(pic, max_size=300)


# def save_multiple_blocks_for_good(block_ids_list, path='', debug=False):
#     debug_report(f"** running save_multiple_blocks_for_good for {block_ids_list}", debug)
#
#     for block_id in block_ids_list:
#         block = get_block(block_id)
#         block.save_block_for_good(debug=debug)
#
#     save_current_data_obj(path=path)


def do_final_results_plot(file_name, block_ids_list, picture=None, debug=False, do_plot=True, fig_size=None, crop_to_mask=True):
    if not do_plot:
        return
    block_ncol = ScanDataObj.get_scan_data(file_name).block_ncol
    debug_report(f'these are the pics: {block_ids_list}', debug)
    data_obj = ScanDataObj.get_scan_data(file_name)
    if not picture:
        picture = {}
        for block_id in block_ids_list:
            block = data_obj.get_block(block_id)
            picture[block_id] = block.plot_block(plot_images=0, label=block_id, with_border=0,
                                                 fig_size=300, crop_to_mask=crop_to_mask, debug=debug)

    for i in range(int(len(block_ids_list) / block_ncol)):
        p1 = picture[block_ids_list[block_ncol * i]]
        p2 = picture[block_ids_list[block_ncol * i + 1]]
        p3 = picture[block_ids_list[block_ncol * i + 2]]
        if block_ncol == 4:
            p4 = picture[block_ids_list[block_ncol * i + 3]]
            top = CommonFunctions.pad_and_concat_images(p1, p2, axis=1)
            bottom = CommonFunctions.pad_and_concat_images(p3, p4, axis=1)
            combined_image = CommonFunctions.pad_and_concat_images(top, bottom, axis=1)
        elif block_ncol == 3:
            temp = CommonFunctions.pad_and_concat_images(p1, p2, axis=1)
            combined_image = CommonFunctions.pad_and_concat_images(temp, p3, axis=1)
        #             row_m_one = CommonFunctions.pad_and_concat_images(p1, p2, axis=1)
        #             combined_image = CommonFunctions.pad_and_concat_images(row_m_one, p3, axis=1)
        CommonFunctions.display_in_console(combined_image, fig_size=fig_size)


#     for j in range(block_ncol*(i+1),len(picture)):
#         print(j)
#         CommonFunctions.display_in_console(picture[j])


def measure_signal_of_blocks(file_name, block_ids_list, sigma1=2, sigma2=2, plot_images=True, debug=False,
                             fg_inc_pixels=2, margin_pixels=2, bg_r=4, debug_blocks_ids=[], image_size=300):
    output_total_counts = np.array([0, 0, 0])
    data_obj = ScanDataObj.get_scan_data(file_name)
    picture = {}
    blocks_dfs = []
    for block_id in block_ids_list:
        try:
            debug_block = True if block_id in debug_blocks_ids else debug
            print(block_id, '....')

            block = data_obj.get_block(block_id)
            # block.add_names_to_clusters(debug=debug)
            block.measure_all_fg_bg_of_block(debug=debug_block, fg_inc_pixels=fg_inc_pixels,
                                             margin_pixels=margin_pixels, bg_r=bg_r, )

            counts = block.calculate_results_in_block(debug=debug_block, sigma1=sigma1, sigma2=sigma2)
            output_total_counts += np.array(counts)
            #         print(output_total_counts)

            blocks_dfs.append(block.give_block_data_df(debug=debug_block))
            #         intensities_dict_list, columns = block.give_block_data_list(debug=debug)
            #         intensities_dict[block_id] = intensities_dict_list

            picture[block_id] = block.plot_block(plot_images=False, debug=debug_block, crop_to_mask=1,
                                                 label=block.block_id, description='cAb:intensities')
        except Exception as e:
            print(f"Exception:\n{e}")
            tb = traceback.format_exc()
            print(f"Traceback:\n{tb}")

    if plot_images:
        do_final_results_plot(file_name=file_name,block_ids_list=block_ids_list, picture=picture, debug=debug_block)

    print(
        f'''delete1: {output_total_counts[0]} ({100 * output_total_counts[0] / output_total_counts[2]:.4f}%), delete2: {output_total_counts[1]} ({100 * output_total_counts[1] / output_total_counts[2]:.4f}%) -> total delete: {100 * (output_total_counts[0] + output_total_counts[1]) / output_total_counts[2]:.2f}%''')
    out_df = pd.concat(blocks_dfs, ignore_index=True)

    #     save_dicts_in_pickle(path=path)
    return out_df, output_total_counts


