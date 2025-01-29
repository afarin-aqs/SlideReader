import pandas as pd
from scipy import ndimage
pd.options.mode.chained_assignment = None  # default='warn'
import inspect
import json
import numpy as np
import hdbscan
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import itertools
import traceback
from functools import partial
import re
import itertools
from copy import deepcopy
from importlib import reload
from Classes import ScanDataObj
reload(ScanDataObj)

#%%
def find_min_max_coords(input_coords_list, min_max_dict):
    min_x = deepcopy(min_max_dict['min_x'])
    max_x = deepcopy(min_max_dict['max_x'])
    min_y = deepcopy(min_max_dict['min_y'])
    max_y = deepcopy(min_max_dict['max_y'])
    for x, y, r in input_coords_list:  # loop over all coords
        if min_x is None or min_x > x:
            min_x = x
        if max_x is None or max_x < x:
            max_x = x
        if min_y is None or min_y > y:
            min_y = y
        if max_y is None or max_y < y:
            max_y = y
    return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}


#%%
color_dict = {}

# %%
def give_variant_from_text(text=""):
    pattern = r'\((.*?)\)'

    # Extract codes in parentheses
    matches = re.findall(pattern, text)
    variants = []
    for match in matches:
        if "Cat#" in match:
            continue
        else:
            variants.append(match)
    # Print the extracted codes
    print(variants)

# %%
def extract_category_numbers(text):
    # Regular expression pattern to match category numbers
    pattern = r'\b\d+-[A-Z]+\d+\b'
    # Extract category numbers
    category_numbers = re.findall(pattern, text)
    return category_numbers


def find_matches(category_numbers, lookup_text):
    # Regular expression pattern to match values with numbers in front
    pattern = r'(\b\d+-[A-Z]+\d+\b)\s+(\d+)'
    # Extract values with numbers in front
    matches = re.findall(pattern, lookup_text)
    # Filter matches based on category numbers
    #     filtered_matches = [(value, number) for value, number in matches if value in category_numbers]
    filtered_matches = [(value, number) if int(number) < 100 else (value, None) for value, number in matches if
                        value in category_numbers]

    return filtered_matches



# %%
def debug_report(string, debug=False):
    if not debug:
        return
    max_len = 20

    caller_frame = inspect.currentframe().f_back
    line_no = caller_frame.f_lineno  # Line number where custom_print was called
    func_name = caller_frame.f_code.co_name  # Name of the function that called custom_print
    text = f"[{func_name[:max_len].ljust(max_len)} - Line {line_no}]"

    caller_caller_frame = caller_frame.f_back if caller_frame else None

    shit = True
    if caller_caller_frame and shit:
        caller_caller_func_name = caller_caller_frame.f_code.co_name
        caller_caller_line_no = caller_caller_frame.f_lineno
        #         if caller_caller_func_name != "<module>":
        text = f"[{caller_caller_func_name[:max_len].ljust(max_len)}({caller_caller_line_no}) - {func_name[:max_len].ljust(max_len)}({line_no})]"

    print(f"{text}: {string}")


# %%
def show(image, plot_images=True):
    if not plot_images:
        return
    img = Image.fromarray(image)
    img.show()


def display_in_console(image, text='', fig_size=None, plot_images=True):
    if not plot_images:
        return
    print('\n', text, end='')
    if fig_size is None:
        fig_size = [7, 7]
    plt.figure(figsize=fig_size)  # Adjust the figure size for better visualization
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off', )
    plt.show()


# %%
def pad_and_concat_images(img1, img2, pad_value=0, sep_width=3, axis=1):  # horizontal: 1, vertical: 0
    if axis == 0:  # Vertical concatenation
        max_width = max(img1.shape[1], img2.shape[1])
        img1_pad = ((0, 0), (0, max_width - img1.shape[1]), (0, 0))
        img2_pad = ((0, 0), (0, max_width - img2.shape[1]), (0, 0))
        separator = np.full((sep_width, max_width, img1.shape[2]), pad_value, dtype=img1.dtype)
    else:  # Horizontal concatenation
        max_height = max(img1.shape[0], img2.shape[0])
        img1_pad = ((0, max_height - img1.shape[0]), (0, 0), (0, 0))
        img2_pad = ((0, max_height - img2.shape[0]), (0, 0), (0, 0))
        separator = np.full((max_height, sep_width, img1.shape[2]), pad_value, dtype=img1.dtype)

    # Apply padding
    img1_padded = np.pad(img1, img1_pad, 'constant', constant_values=pad_value)
    img2_padded = np.pad(img2, img2_pad, 'constant', constant_values=pad_value)

    combined_image = np.concatenate((img1_padded, separator, img2_padded), axis=axis)

    return combined_image


# %%
def give_scaled_log_image(input_image, debug=False):
    debug_report(f'input_image range: {int(np.min(input_image))} - {int(np.max(input_image))} ({input_image.dtype})', debug)

    if np.min(input_image) == 0:
        input_image[input_image == 0] = 1

    logged_image = np.log10(input_image).astype(np.uint8)
    debug_report(f'logged_image range: '
                 f'{int(np.min(logged_image))} - {int(np.max(logged_image))} '
                 f'({logged_image.dtype})', debug)

    scaled_logged_image = cv2.normalize(logged_image, None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    debug_report(f'scaled_logged_image range: '
                 f'{int(np.min(scaled_logged_image))} - {int(np.max(scaled_logged_image))} '
                 f'({scaled_logged_image.dtype})', debug)

    return scaled_logged_image


def load_image(file_name, path,  rotation=0, debug=False, plot_images=False):
    input_path = path + '/' + file_name + '.tif'
    debug_report(f'running "load_image" for {input_path}', debug)
    try:
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if rotation == 180:
            # image = np.array(image.transpose(Image.ROTATE_180))
            image = cv2.rotate(image, cv2.ROTATE_180)

        debug_report(f'image range: {int(np.min(image))} - {int(np.max(image))}', debug)
        scaled_logged_image = give_scaled_log_image(image, debug)

        if plot_images:
            show(scaled_logged_image)

        ScanDataObj.add_to_images_dict(file_name=file_name, dict_key='file_image', dict_value=image)
        ScanDataObj.add_to_images_dict(file_name=file_name, dict_key='file_scaled_image', dict_value=scaled_logged_image)
        return image, scaled_logged_image
    except Exception as e:
        print(f"Exception:\n{e}")
        tb = traceback.format_exc()
        print(f"Traceback:\n{tb}")
        return None, None


# %%
def image_preprocessing(input_image, params=None, debug=False, plot_images=False):
    if not params:
        params = {'contrast_thr': 500, 'blur_kernel_size': 3, 'canny_edge_thr1': 300, 'canny_edge_thr2': 700}
        debug_report(f"""Defaulting preprocessing params to {params}""", debug)

    debug_report(f'running image_preprocessing function with params = {params}', debug)
    output_image = deepcopy(input_image)

    # Contrast Thresholding
    if params['contrast_thr']:
        ret, output_image = cv2.threshold(output_image, params['contrast_thr'], 255, 1)
        output_image = output_image.astype(np.uint8)
        debug_report(f'did the contrast threshholding', debug)

    # Blurring
    if params['blur_kernel_size']:
        output_image = cv2.GaussianBlur(output_image, (params['blur_kernel_size'], params['blur_kernel_size']), 0)
        debug_report(f'did the blurring', debug)

    # Canny Edge Detection
    if params['canny_edge_thr1']:
        output_image = cv2.Canny(output_image, threshold1=params['canny_edge_thr1'],
                                 threshold2=params['canny_edge_thr2'])
        debug_report(f'did the canny edge detection', debug)

    display_in_console(image=output_image, plot_images=plot_images)
    return output_image


# %%
## Circle Finding
def show_circles_on_plot(input_image, circles_vec=None, debug=False, fig_size=None):
    debug_report(f'** running show_circles_on_plot with {len(circles_vec)} circles', debug)
    if circles_vec is None:
        circles_vec = []
    temp_plot_image = give_scaled_log_image(input_image, debug).astype(np.uint8)
    for x, y, r in circles_vec:
        cv2.circle(temp_plot_image, (x, y), r, (255, 0, 0), 2)
    display_in_console(image=temp_plot_image, fig_size=fig_size)


def circle_detection(input_image, detection_params, preprocess_params=None, debug=False, plot_images=False):
    try:
        method = detection_params['method_name']

        # Check/Do PreProcessing
        if preprocess_params:
            preprocessed_image = image_preprocessing(input_image, preprocess_params, debug, plot_images)
        else:
            preprocessed_image = deepcopy(input_image)

        if 'hough' in method.lower():
            sorted_circles = find_circles_with_hough_transform(preprocessed_image, detection_params, debug, plot_images)
        elif 'contour' in method.lower():
            sorted_circles = find_circles_with_contour_detection(preprocessed_image, detection_params)
        debug_report(f'found {len(sorted_circles)} circles', debug)
        return sorted_circles
    except Exception as e:
        print(f"Exception:\n{e}")
        tb = traceback.format_exc()
        print(f"Traceback:\n{tb}")
    return []


def find_circles_with_hough_transform(input_image, params, debug, plot_images):
    debug_report(f'running find_circles_with_hough_transform function', debug)

    # circle detection:
    circles = cv2.HoughCircles(
        input_image,
        cv2.HOUGH_GRADIENT,
        dp=params['dp'],
        minDist=params['minDist'],
        param1=params['param1'],
        param2=params['param2'],
        minRadius=params['minRadius'],
        maxRadius=params['maxRadius']
    )

    if circles is None:
        debug_report("No circles were detected.", debug)
        return []

    if 'r_plus' in params:
        r_plus = params['r_plus']
    else:
        r_plus = 0

    circles = np.uint16(np.around(circles))  # [center_x, center_y, radius]
    circles = circles[0]
    debug_report(f'** Detected total of {len(circles)} circles', debug)
    sorted_indices = np.lexsort((circles[:, 0], circles[:, 1]))
    sorted_circles = circles[sorted_indices]
    sorted_circles = [np.append(item[:-1], item[-1] + r_plus) for item in sorted_circles]

    debug_report(f'found {len(sorted_circles)} circles with cv2.HoughCircles', debug)
    debug_report(f'**added {r_plus} to the radius of each circle**', debug & r_plus > 0)

    if not plot_images:
        return sorted_circles

    show_circles_on_plot(input_image, circles_vec=sorted_circles, debug=debug)
    return sorted_circles


def find_circles_with_contour_detection(input_image, params, plot_images=False, debug=False):
    debug_report(f'running find_circles_with_contour_detection function', debug)

    # initiating
    temp_image = deepcopy(input_image)

    if 'r_plus' in params:
        r_plus = params['r_plus']
    else:
        r_plus = 0

        # contour detection:
    contours, _ = cv2.findContours(np.array(temp_image, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []

    for contour in contours:
        if len(contour) < params['min_arc_length']:
            continue
        (x, y), r = cv2.minEnclosingCircle(contour)

        if r < params['minRadius'] or r > params['maxRadius']:
            continue

        circles.append(np.array([np.ceil(x), np.ceil(y), np.ceil(r + r_plus)], dtype=np.int16))
        debug_report(f'Circle detected: {(x, y, r)}', debug)

    if circles is None:
        debug_report("No circles were detected.", debug)
        return []

    if 'r_plus' in params:
        r_plus = params['r_plus']
    else:
        r_plus = 0

    if circles == []:
        debug_report("No circles were detected.", debug)
        return []

    circles = np.array(circles)
    debug_report(f'** Detected total of {len(circles)} circles', debug)

    sorted_indices = np.lexsort((circles[:, 0], circles[:, 1]))
    sorted_circles = circles[sorted_indices]
    sorted_circles = [np.append(item[:-1], item[-1] + r_plus) for item in sorted_circles]

    debug_report(f'found {len(sorted_circles)} circles with cv2.HoughCircles', debug)
    debug_report(f'**added {r_plus} to the radius of each circle**', debug & r_plus > 0)

    if not plot_images:
        return sorted_circles

    show_circles_on_plot(input_image, circles_vec=sorted_circles, debug=debug)
    return sorted_circles


# %%
# Parameter Optimization
# def range_dict_check(range_dict):
#     for k, v in range_dict.items():
#         try:
#             iter(v)
#             continue
#         except TypeError:
#             if type(v) == str:
#                 continue
#             v = (v,)
#         range_dict[k] = v
#     return range_dict

def how_many_cluster_circles_is_here(input_image, preprocess_params, clustering_params,
                                    circle_finding_params, debug=False, plot_images=False, max_num_of_circles=1000):
    input_image = deepcopy(input_image)
    # debug_report(f'{preprocess_params} & {circle_finding_params}', debug)
    found_circles = circle_detection(
        input_image, detection_params=circle_finding_params, preprocess_params=preprocess_params
    )
    if not found_circles or found_circles==[] or len(found_circles) > max_num_of_circles:
        return None, None

    predicted_clusters_ids, colored_image = DBSCAN_clustering(
        found_circles, input_image,params=clustering_params, return_colored_img_too=True
    )
    found_cluster_circles_counts = len([x for x in predicted_clusters_ids if x != -1]) * len({x for x in predicted_clusters_ids if x != -1})
    return found_cluster_circles_counts, colored_image


def mix_and_make_iterable(optimization_params, default_values):
    output_dict = deepcopy(default_values)
    for k, v in default_values.items():
        if k in optimization_params:
            output_dict[k] = optimization_params[k]
            continue
        try:
            iter(v)
            continue
        except TypeError:
            if type(v) == str:
                continue
            else:
                output_dict[k] = (v,)
    return output_dict


def do_parameter_optimization(input_image, init_params=None, max_num_of_circles=200,
                              search_step=1, debug=False, plot_images=False):
    input_image = deepcopy(input_image)

    init_preprocess_params = init_params['pp']
    init_circle_finding_params = init_params['cf']
    init_clustering_params = init_params['cl']

    debug_report(f'init_params={init_params}',debug)
    c = search_step

    optimization_params = {
        'pp': {
            'blur_kernel_size': [max(1,init_preprocess_params['blur_kernel_size'] + x + (0 if (init_preprocess_params['blur_kernel_size'] + x) % 2 == 1 else 1)) for x in [-9//c, 0, 9//c]],
            'contrast_thr': [max(1,init_preprocess_params['contrast_thr'] + x) for x in [-100//c, 0, 100//c]],
            'canny_edge_thr1': [max(1,init_preprocess_params['canny_edge_thr1'] + x) for x in [-20//c, 0, 20//c]],
            'canny_edge_thr2': [max(1,init_preprocess_params['canny_edge_thr2'] + x) for x in [-20//c, 0, 20//c]],
        },
        'cf': {
            'dp': [max(1,init_circle_finding_params['dp'] + x) for x in [-0.3//c, 0, 0.3//c]],
            'param1': [max(1,init_circle_finding_params['param1'] + x) for x in [-15//c, 0, 15//c]],
            'param2': [max(1,init_circle_finding_params['param2'] + x) for x in [-15//c, 0, 15//c]],
        }
    }

    debug_report(optimization_params,debug)
    # todo: clean up here ...
    pp_combos = list(itertools.product(*optimization_params['pp'].values()))
    cf_combos = list(itertools.product(*optimization_params['cf'].values()))
    total_steps = len(pp_combos) * len(cf_combos)
    print(f'pp_combinations are {len(pp_combos)} and cf_combinations are {len(cf_combos)} => total combinations={total_steps}')
    print('Checked Combinations:')
    max_cluster_circles_count = 0
    max_num_clusters = 0
    best_colored_image = None
    final_optimized_params = {}
    step = 0
    for one_pp in pp_combos:
        for one_cf in cf_combos:
            debug_report(f'one_pp={one_pp}, one_cf={one_cf}', debug)
            step += 1
            if step % 100 == 0:
                print(f'{step}', end=', ')
            pp_step_dict = dict(zip(optimization_params['pp'].keys(), one_pp))
            cf_step_dict = dict(zip(optimization_params['cf'].keys(), one_cf))

            debug_report(f'pp_step_dict={pp_step_dict}, cf_step_dict={cf_step_dict}', debug)
            cf_step_dict.update(
                {k: v for k, v in init_circle_finding_params.items() if k not in optimization_params['cf']})

            number_of_cluster_circles, colored_image = how_many_cluster_circles_is_here(
                input_image=input_image, preprocess_params=pp_step_dict,
                clustering_params=init_clustering_params,
                circle_finding_params=cf_step_dict,
                debug=debug, plot_images=plot_images,
                max_num_of_circles=max_num_of_circles)

            if not number_of_cluster_circles:
                # debug_report(f"@ step {step}: 0 circles were detected.", debug)
                continue

            if number_of_cluster_circles <= max_cluster_circles_count or number_of_cluster_circles > max_num_of_circles:
                continue
            debug_report(f'@ step {step}: max_cluster_circles_count={max_cluster_circles_count} '
                         f'and number_of_cluster_circles={number_of_cluster_circles}', debug)

            max_cluster_circles_count = number_of_cluster_circles
            final_optimized_params = {
                'pp': deepcopy(pp_step_dict),
                'cf': deepcopy(cf_step_dict),
            }
            best_colored_image = deepcopy(colored_image)
            if plot_images or debug:
                display_in_console(best_colored_image)


    debug_report(f'in the end, optimized_params={final_optimized_params}', debug)
    display_in_console(best_colored_image)
    final_optimized_params['cl'] = init_clustering_params
    return final_optimized_params



# %%
# Parameter Optimization for PreProcessing:

# def preprocess_param_optimization(input_image, detection_params, preprocess_params_ranges_dict,
#                                   near_optimal_count=1000, near_optimal_delta=50, debug=False, plot_images=False, ):
#     try:
#         preprocess_params_ranges_dict = range_dict_check(preprocess_params_ranges_dict)
#
#         optimal_circles_found = []
#
#         param_iterations = itertools.product(*preprocess_params_ranges_dict.values())
#         print(f'There are total of {len(list(deepcopy(param_iterations)))} iterations to check')
#
#         for step, step_param_values in enumerate(param_iterations):
#             step_preprocess_params_dict = dict(zip(preprocess_params_ranges_dict.keys(), step_param_values))
#             debug_report(f'@ step {step}:\step_preprocess_params_dict={step_preprocess_params_dict}', debug)
#
#             found_circles = circle_detection(input_image, detection_params, step_preprocess_params_dict)
#             N = len(found_circles)
#
#             if np.abs(N - near_optimal_count) > near_optimal_delta:
#                 continue
#
#             if len(found_circles) <= len(optimal_circles_found):
#                 continue
#
#             optimal_circles_found = deepcopy(found_circles)
#             optimized_params = deepcopy(step_preprocess_params_dict)
#             print(f"@ step {step}: ", end='')
#             print(", ".join(f"{k}: {v}" for k, v in optimized_params.items()), end='')
#             print(f' ---> found {len(optimal_circles_found)} circles')
#
#             temp_plot_image = give_scaled_log_image(input_image, debug).astype(np.uint8)
#             if not plot_images:
#                 continue
#             show_circles_on_plot(temp_plot_image, circles_vec=optimal_circles_found, debug=debug)
#
#         show_circles_on_plot(temp_plot_image, circles_vec=optimal_circles_found, debug=debug)
#     except Exception as e:
#         tb = traceback.format_exc()
#         print(tb)
#     return optimized_params


# %%
# Clustering Spots

def make_3D_image(input_image, astype=np.uint8):
    output_image = deepcopy(input_image)
    output_image = np.stack((output_image,) * 3, axis=-1)
    output_image = output_image.astype(astype)
    return output_image


def give_me_color_for_cluster_id(cluster_id):
    global color_dict

    if cluster_id in color_dict.keys():
        return tuple(color_dict[cluster_id])
    else:
        color = tuple(np.random.randint(90, 250, size=3))
        color = [int(value) for value in color]
        color_dict[cluster_id] = color
        return tuple(color)


def distance_metric(circle1, circle2, params):
    x1, y1, r1 = circle1.astype(int)
    x2, y2, r2 = circle2.astype(int)
    # The default and right way of printing is horizantal! -> distance = dx^3 + dy^5
    if 'x_power' not in params:
        params['x_power'] = 3
    if 'y_power' not in params:
        params['y_power'] = 5

    extra_y_cost = 0
    if 'extra_y_cost' in params and params['extra_y_cost'] is True:
        extra_y_cost = -(y1.astype(int)+y2.astype(int)).astype(int)

    delta_x = x1.astype(int) - x2.astype(int)
    delta_y = y1.astype(int) - y2.astype(int)

    cost = np.sqrt(np.abs(np.abs(delta_x) ** params['x_power'] + np.abs(delta_y) ** params['y_power']+extra_y_cost))

    #     print(circle1,circle2,cost)
    return cost


def DBSCAN_clustering(sorted_circles, input_image, params=None, plot_images=False,
                      debug=False, debug_clusters_ids=None, fig_size=None, return_colored_img_too=False):

    debug_report(f'\n** running "DBSCAN_clustering"', debug)

    if params is None:
        params = {}

    if not debug_clusters_ids:
        debug_clusters_ids=[]

    # this is the cool part!! :D
    custom_metric = partial(distance_metric, params=params)
    clustering_model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric=custom_metric)

    # Fit the DBSCAN model to the circle data
    predicted_clusters_ids = clustering_model.fit_predict(sorted_circles)

    total_found_circles = len(sorted_circles)
    clustered_circles = predicted_clusters_ids[predicted_clusters_ids != -1]
    unique_cluster_ids = np.unique(clustered_circles)
    performance = 100 * len(clustered_circles) / total_found_circles
    debug_report(
        f'found {len(unique_cluster_ids)} clusters, with total of {len(clustered_circles)} circles ({performance:.0f}%)',
        debug)

    #     if not plot_images and not debug:
    #         return clustered_circles

    colored_image = make_3D_image(input_image)
    if debug_clusters_ids or debug:
        debug_colored_image = make_3D_image(input_image)
    clusters_ids_list = []
    # debug_plot_coords = {}
    plot_coords = [np.inf,np.inf,0,0]
    for index, cluster_id in enumerate(predicted_clusters_ids):
        debug_cluster = debug or cluster_id in debug_clusters_ids
        spot_coords = [int(x) for x in sorted_circles[index]]
        x, y, r = spot_coords
        cv2.circle(colored_image, (x, y), r, (0, 0, 0), 3)

        if cluster_id == -1:
            debug_report(f'this dude ({x, y, r}) is an outlier!', debug_cluster)
            continue

        debug_report(f'this circle ({x, y, r}) is labeled as {cluster_id}', debug_cluster)
        color = give_me_color_for_cluster_id(cluster_id)
        cv2.circle(colored_image, (x, y), r, color, 2)
        if cluster_id not in clusters_ids_list:
            text_size = cv2.getTextSize(str(cluster_id), cv2.FONT_HERSHEY_SIMPLEX, 0.1, 1)[0]
            text_x = max(x,25) - text_size[0] // 2 - 10
            text_y = max(y,25) + text_size[1] // 2 - 10

            cv2.putText(colored_image, str(cluster_id), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)
            clusters_ids_list.append(cluster_id)
        if debug_cluster:
            # print('here for cluster:', cluster_id, (x,y,r))
            if x < plot_coords[0]:
                plot_coords[0] = x
                # print(f'updated plot_coords: {plot_coords}')
            if y < plot_coords[1]:
                plot_coords[1] = y
                # print(f'updated plot_coords: {plot_coords}')
            if x > plot_coords[2]:
                plot_coords[2] = x
                # print(f'updated plot_coords: {plot_coords}')
            if y > plot_coords[3]:
                plot_coords[3] = y
                # print(f'updated plot_coords: {plot_coords}')
            cv2.circle(debug_colored_image, (x, y), r, color, 2)
            text_size = cv2.getTextSize(str(cluster_id), cv2.FONT_HERSHEY_SIMPLEX, 0.1, 1)[0]
            text_x = max(x, 25) - text_size[0] // 2 - 10
            text_y = max(y, 25) + text_size[1] // 2 - 10
            # print('test x,y:',text_x, text_y)
            cv2.putText(debug_colored_image, str(index), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)
            # print(f'added {(x,y,r)} with index={index} to debug_colored_image')
            # if cluster_id in debug_plot_coords:
            #     debug_plot_coords[cluster_id].append((x, y, r))
            # else:
            #     debug_plot_coords[cluster_id] = [(x, y, r)]

    if plot_images:
        display_in_console(colored_image, plot_images=plot_images, fig_size=fig_size)
    if debug_clusters_ids:
        # print(debug_colored_image.shape, plot_coords)
        pad = 10
        i = 1
        j = 0
        # print(1, debug_colored_image.shape, 'and coords are', plot_coords)
        # print(f'[{plot_coords[i]}-{pad}:{plot_coords[i+2]}+{pad}, {plot_coords[j]}-{pad}:{plot_coords[j+2]}+{pad}]')
        # print(f'=== [{plot_coords[i]-pad}:{plot_coords[i+2]+pad}], {plot_coords[j]-pad}:{plot_coords[j+2]+pad}')
        img = debug_colored_image[plot_coords[i]-pad:plot_coords[i+2]+pad,plot_coords[j]-pad:plot_coords[j+2]+pad]
        # print(2, img.shape)
        show(img, plot_images=True)
    if return_colored_img_too:
        return predicted_clusters_ids, colored_image
    return predicted_clusters_ids


# %%
# def plot_2_blocks(block_id_1, block_id_2, debug=False):
#     image1 = get_block(block_id_1).plot_block(really_plot=False, debug=debug)
#     image2 = get_block(block_id_2).plot_block(really_plot=False, debug=debug)

#     combined_image = np.concatenate((image1, image2), axis=1)
#     combined_image = Image.fromarray(combined_image)
#     display(combined_image)
# %%
def outlier_removal(input_list, sigma=1):
    filtered_array = np.array([x for x in input_list if x != ''])

    avg = np.mean(filtered_array)
    std = np.std(filtered_array)

    lower_bound = np.max([avg - (sigma * std), 0])
    upper_bound = avg + (sigma * std)
    #     print(avg, std, lower_bound,upper_bound)

    delete_indices = []
    output_list = []
    for element in input_list:
        if element == '':
            output_list.append('')
        elif element <= upper_bound and element >= lower_bound:
            output_list.append(element)
        else:
            output_list.append('')
            delete_indices.append(list(np.where(filtered_array == element)[0])[0])

    return output_list, delete_indices


# import numpy as np
# a = [1,2,4,6,1000,1200]
# dd,dddd = outlier_removal(a)
# print(dddd)
# %%

# %%

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")


# def save_clusters_dict(filename):
#     with open(filename, 'w') as file:
#         temp = {}
#         for cluster_id, cluster in clusters_dict.items():
#             attributes = {}
#             for attribute_name, attribute_value in cluster.__dict__.items():
#                 if attribute_name in ['fg_mask_list', 'bg_mask_list']:
#                     continue
#                 attributes[attribute_name] = attribute_value
#             temp[int(cluster_id)] = attributes
#         json.dump(temp, file, default=numpy_to_python)
#
#
# def save_blocks_dict(filename):
#     with open(filename, 'w') as file:
#         temp = {}
#         for block_id, block in blocks_dict.items():
#             attributes = {}
#             for attribute_name, attribute_value in block.__dict__.items():
#                 if attribute_name in ['image', 'log_image', 'neg_fg_mask']:
#                     continue
#                 attributes[attribute_name] = attribute_value
#             temp[block_id] = attributes
#         json.dump(temp, file, default=numpy_to_python)


# def load_clusters_dict(filename):
#     with open(filename, 'r') as file:
#         my_dict = {}
#         x = json.load(file)
#         for name, attributes in x.items():
#             my_dict[int(name)] = Cluster(**attributes)
#     return my_dict
#
#
# def load_blocks_dict(filename):
#     with open(filename, 'r') as file:
#         my_dict = {}
#         x = json.load(file)
#         for name, attributes in x.items():
#             my_dict[name] = Block(**attributes)
#     return my_dict
#

# save_blocks_dict(f'blocksdict{c}.csv')
# save_clusters_dict(f'clustersdict{c}.csv')
# my_dict = load_blocks_dict(f'blocksdict{c}.csv')
# print("\n".join(["{}  |  {}".format(name, instance) for name, instance in sorted(my_dict.items())]))
# %%
def print_inside_dict_shapes(input_dict=None):
    shapes = {}

    if input_dict is None:
        print('input_dict is None!')
        return shapes

    for key, value in input_dict.items():
        if value is None:
            shapes[key] = None

        elif isinstance(value, dict):
            shapes[key] = print_inside_dict_shapes(value)

        elif isinstance(value, pd.DataFrame):
            shapes[key] = value.shape

        elif isinstance(value, list):
            shapes[key] = len(value)

    return shapes
# %%
def optimized_spots_coords(input_image, coords_list, fg_inc_pixels=1, search_vec=None, avg_r=25, debug=False):
    if search_vec is None:
        search_vec = [-2, -1, 0, 1, 2]
        debug_report(f'search_vec is {search_vec}', debug)

    input_image = deepcopy(input_image)
    debug_report(f'input image is {input_image.shape}',debug)

    foreground = np.zeros(input_image.shape)

    debug_report(f'Before: coords_list={coords_list}', debug)

    i = 0
    highest_intensities_per_spot = []
    best_coords_for_spots = []
    for x, y, r in coords_list:
        spot_best = {'intensity': 0, 'coords': [0, 0, 0]}
        for dx in search_vec:
            for dy in search_vec:
                r_vec = [0, 1, 2] if r <= avg_r else [0,-1,-2]
                for dr in r_vec:
                    nx = x + dx
                    ny = y + dy
                    nr = r + dr + fg_inc_pixels


                    spot_fg = np.zeros(input_image.shape)
                    # spot_fg = deepcopy(foreground)
                    cv2.circle(spot_fg, (nx, ny), nr, 1, thickness=-1)
                    fg_label, _ = ndimage.label(spot_fg)
                    fg_mean = ndimage.mean(input_image, fg_label)
                    debug_report(f'(x,y,r)={(x,y,r)} - fg_mean={fg_mean}', debug)
                    if fg_mean > spot_best['intensity']: #checkme whats happening here?
                        spot_best['intensity'] = fg_mean
                        spot_best['coords'] = [nx, ny, nr]
                        debug_report(f"fg_mean={fg_mean}>  spot_best['intensity']={spot_best['intensity']}", debug)
        highest_intensities_per_spot.append(spot_best['intensity'])
        best_coords_for_spots.append(spot_best['coords'])
        i += 1
    new_coords_list = deepcopy(best_coords_for_spots)
    debug_report(f"""new_coords_list={new_coords_list}""", debug)
    return new_coords_list, highest_intensities_per_spot


#%%
def optimize_the_params(file_name,  how_many_times=1, input_image=None,
                        max_num_of_circles=200, plot_images=False, debug=False,):

    scan_data = ScanDataObj.get_scan_data(file_name=file_name)
    if input_image is None:
        input_image = ScanDataObj.get_image_from_dict(file_name=file_name, dict_key='file_image')

    new_params = {}
    working_params = {
        'pp': scan_data.preprocess_params,
        'cf': scan_data.circle_finding_params_hough,
        'cl': scan_data.clustering_params_DBSCAN
    }
    search_step = 1

    for t in range(how_many_times):
        search_step *= (t+1)
        new_params = do_parameter_optimization(
            input_image=input_image,
            debug=debug, plot_images=plot_images,
            max_num_of_circles=max_num_of_circles,
            search_step=search_step,
            init_params=working_params,
        )
        working_params = deepcopy(new_params)

    for k, v in new_params.items():
        print(f'Updated params to this:\nkey: {k}\n{v}\n')
    return new_params


#%%
def test_current_parameters(input_image, file_name, fig_size=None, debug=False):
    if fig_size is None:
        fig_size = [10,10]

    scan_data = ScanDataObj.get_scan_data(file_name=file_name)
    test_sorted_circles = circle_detection(
        input_image=input_image,
        detection_params=scan_data.circle_finding_params_hough,
        preprocess_params=scan_data.preprocess_params,
        debug=debug,
        plot_images=False
    )
    if debug:
        show_circles_on_plot(input_image, fig_size=fig_size, circles_vec=test_sorted_circles, debug=debug)

    test_predicted_clusters_ids = DBSCAN_clustering(
        sorted_circles=test_sorted_circles,
        input_image=input_image,
        params=scan_data.clustering_params_DBSCAN,
        plot_images=True,
        debug=debug,
        debug_clusters_ids=[],
        fig_size=fig_size,
    )
    return



#%%
def do_initial_circle_finding(file_name, debug=False, plot_images=False):
    scan_data = ScanDataObj.get_scan_data(file_name=file_name)
    image = ScanDataObj.get_image_from_dict(file_name=file_name, dict_key='file_image')
    scaled_image = ScanDataObj.get_image_from_dict(file_name=file_name, dict_key='file_scaled_image')

    print('doing the circle finding...', end=' ')
    sorted_circles = circle_detection(
        image,
        detection_params=scan_data.circle_finding_params_hough,
        preprocess_params=scan_data.preprocess_params,
        debug=debug,
        plot_images=plot_images
    )
    print('Done!')
    scan_data.sorted_circles = sorted_circles

    print('doing the clustering...', end=' ')
    predicted_clusters_ids = DBSCAN_clustering(
        sorted_circles=sorted_circles,
        input_image=scaled_image,
        params=scan_data.clustering_params_DBSCAN,
        plot_images=plot_images,
        debug=debug,
    )
    print('Done!')

    scan_data.predicted_clusters_ids = predicted_clusters_ids
    return



