import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


# def step_wise():
#     # Creating a DataFrame from the provided data
#     data = {
#         "Category": [4, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 3, 1,
#                      1, 2, 3, 4, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1,
#                      2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 3, 1, 1, 2, 3],
#         "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 1, 1, 1, 1, 1, 40, 40, 40, 40, 1, 1, 1, 1, 1, 40, 40, 40, 40, 1, 1, 1, 1,
#                 1, 40, 40, 40, 40, 1, 1, 1, 1, 1, 40, 40, 40, 40, 1, 1, 1, 1, 1, 40, 40, 40, 40, 1, 1, 1, 1, 1, 40, 40,
#                 40, 40, 1, 1, 1, 1, 1, 40, 40, 40, 40, 1, 1, 1, 1, 1, 40, 40, 40, 40, 1, 1, 1, 1, 1, 40, 40, 40, 40],
#         "Neck": [20, 21, 7, 10, 9, 19, 18, 18, 22, 13, 19, 4, 19, 7, 19, 23, 16, 2, 21, 22, 22, 22, 19, 21, 17, 22, 26,
#                  11, 19, 10, 10, 18, 21, 18, 8],
#         "Elbow": [16, 0, 75, 83, 1, 76, 74, 64, 69, 6, 2, 74, 80, 12, 72, 60, 19, 0, 68, 81, 77, 87, 69, 72, 75, 73, 72,
#                   83, 72, 67, 84, 79, 178, 64, 156],
#         "Bend": [6, 2, 2, 7, 1, 8, 0, 3, 0, 0, 3, 0, 1, 0, 7, 0, 0, 0, 4, 8, 4, 4, 0, 7, 5, 4, 3, 0, 9, 0, 0, 7, 3, 3,
#                  0, 0, 0, 0],
#         "Shoulder": [65, 63, 44, 41, 57, 55, 56, 45, 48, 51, 0, 15, 18, 0, 50, 19, 0, 15, 45, 56, 45, 49, 57, 50, 17,
#                      51, 32, 80, 53, 37, 44, 53, 53, 78, 45, 71],
#         "Torso": [180, 183, 188, 179, 167, 187, 179, 188, 182, 176, 179, 187, 176, 183, 182, 171, 177, 164, 180, 180,
#                   180, 180, 188, 182, 173, 178, 179, 178, 182, 192, 181, 181, 181, 183, 188],
#         "Knee": [176, 193, 194, 179, 186, 190, 177, 187, 183, 174, 171, 175, 166, 181, 177, 169, 172, 145, 170, 173,
#                  182, 174, 178, 177, 171, 169, 165, 183, 173, 189, 172, 177, 177, 172, 177],
#         "Foot": [182, 215, 200, 181, 176, 201, 194, 206, 198, 174, 170, 155, 179, 193, 186, 166, 182, 165, 161, 172,
#                  194, 172, 211, 166, 166, 171, 161, 183, 211, 219, 183, 179, 188, 206, 168],
#         "step": ['step1'] * 9 + ['step2'] * 9 + ['step3'] * 9 + ['step4'] * 9
#     }
#
#     df = pd.DataFrame(data)
#
#     # Plotting comparison including per step
#     plt.figure(figsize=(12, 8))
#
#     for step in df['step'].unique():
#         step_data = df[df['step'] == step]
#         for category in step_data['Category'].unique():
#             category_data = step_data[step_data['Category'] == category]
#             plt.plot(category_data['Day'], category_data['Neck'], label=f'Category {category} - {step}', marker='o')
#
#     plt.xlabel('Day')
#     plt.ylabel('Neck')
#     plt.title('Comparison of Categories by Step')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def category_wise():
    # Creating a DataFrame from the provided data
    data = {
        "Category": [4, 1, 1, 2, 3, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40],
        "Neck_step1": [20, 21, 7, 10, 9, 19, 18, 18, 22],
        "Elbow_step1": [16, 0, 75, 83, 1, 76, 74, 64, 69],
        "Bend_step1": [6, 2, 2, 7, 1, 8, 0, 3, 0],
        "Shoulder_step1": [65, 63, 44, 41, 57, 55, 56, 45, 48],
        "Torso_step1": [180, 183, 188, 179, 167, 187, 179, 188, 182],
        "Knee_step1": [176, 193, 194, 179, 186, 190, 177, 187, 183],
        "Foot_step1": [182, 215, 200, 181, 176, 201, 194, 206, 198]
    }

    df = pd.DataFrame(data)

    # Filtering data for categories 1, 2, and 3
    cat_1 = df[df['Category'] == 1]
    cat_2 = df[df['Category'] == 2]
    cat_3 = df[df['Category'] == 3]
    cat_4 = df[df['Category'] == 4]

    # Calculating means for each category and each day
    cat_1_means = cat_1.groupby('Day').mean().reset_index()
    cat_2_means = cat_2.groupby('Day').mean().reset_index()
    cat_3_means = cat_3.groupby('Day').mean().reset_index()
    cat_4_means = cat_4.groupby('Day').mean().reset_index()

    # Plotting comparison between categories 1, 2, and 3 with category 4
    plt.figure(figsize=(10, 6))
    plt.plot(cat_4_means['Day'], cat_4_means['Neck_step1'], label='Category 4', marker='o')
    plt.plot(cat_1_means['Day'], cat_1_means['Neck_step1'], label='Category 1', marker='o')
    plt.plot(cat_2_means['Day'], cat_2_means['Neck_step1'], label='Category 2', marker='o')
    plt.plot(cat_3_means['Day'], cat_3_means['Neck_step1'], label='Category 3', marker='o')

    plt.xlabel('Day')
    plt.ylabel('Mean Values')
    plt.title('Comparison of Categories 1, 2, 3 with Category 4 (Neck_step1)')
    plt.legend()
    plt.grid(True)
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # step1_min_data = {
    #     "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
    #     "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
    #     "Neck": [20, 21, 7, 10, 9, 20, 19, 18, 18, 22],
    #     "Elbow": [16, 0, 75, 83, 1, 16, 76, 74, 64, 69],
    #     "Bend": [6, 2, 2, 7, 1, 6, 8, 0, 3, 0],
    #     "Shoulder": [65, 63, 44, 41, 57, 65, 55, 56, 45, 48],
    #     "Torso": [180, 183, 188, 179, 167, 180, 187, 179, 188, 182],
    #     "Knee": [176, 193, 194, 179, 186, 176, 190, 177, 187, 183],
    #     "Foot": [182, 215, 200, 181, 176, 182, 201, 194, 206, 198]
    # }
    #
    # step2_min_data = {
    #     "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
    #     "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
    #     "Neck": [13, 19, 4, 19, 7, 13, 19, 23, 16, 2],
    #     "Elbow": [6, 2, 74, 80, 12, 6, 72, 60, 19, 0],
    #     "Bend": [0, 3, 0, 1, 0, 0, 7, 0, 0, 0],
    #     "Shoulder": [1, 0, 15, 18, 0, 1, 50, 19, 0, 15],
    #     "Torso": [179, 179, 187, 176, 183, 179, 183, 171, 177, 164],
    #     "Knee": [174, 171, 175, 166, 181, 174, 177, 169, 172, 145],
    #     "Foot": [174, 170, 155, 179, 193, 174, 186, 166, 182, 165]
    # }
    #
    # step3_min_data = {
    #     "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
    #     "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
    #     "Neck": [21, 22, 22, 22, 19, 21, 19, 17, 22, 26],
    #     "Elbow": [68, 81, 77, 87, 69, 68, 72, 75, 73, 72],
    #     "Bend": [4, 8, 4, 4, 0, 4, 9, 5, 4, 3],
    #     "Shoulder": [51, 56, 45, 49, 57, 51, 50, 17, 51, 32],
    #     "Torso": [176, 180, 180, 180, 188, 176, 182, 173, 178, 179],
    #     "Knee": [170, 173, 182, 174, 178, 170, 177, 171, 172, 169],
    #     "Foot": [161, 172, 194, 172, 211, 161, 177, 166, 171, 161]
    # }
    #
    # step4_min_data = {
    #     "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
    #     "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
    #     "Neck": [11, 19, 10, 10, 18, 11, 21, 18, 8, 14],
    #     "Elbow": [83, 72, 67, 84, 79, 83, 178, 64, 156, 158],
    #     "Bend": [0, 4, 0, 0, 7, 0, 3, 3, 0, 0],
    #     "Shoulder": [80, 53, 37, 44, 53, 80, 78, 45, 71, 19],
    #     "Torso": [178, 182, 192, 176, 181, 178, 181, 188, 181, 184],
    #     "Knee": [175, 173, 189, 173, 172, 175, 177, 187, 175, 165],
    #     "Foot": [183, 211, 219, 183, 179, 183, 181, 206, 168, 149]
    # }
    #
    # step1_max_data = {
    #     "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
    #     "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
    #     "Neck": [31, 44, 31, 32, 68, 31, 27, 44, 36, 41],
    #     "Elbow": [38, 359, 109, 115, 359, 38, 88, 90, 102, 90],
    #     "Bend": [14, 11, 22, 28, 12, 14, 15, 29, 16, 27],
    #     "Shoulder": [76, 265, 60, 68, 83, 76, 62, 55, 57, 292],
    #     "Torso": [279, 250, 254, 253, 268, 279, 259, 246, 216, 240],
    #     "Knee": [312, 299, 323, 282, 312, 312, 298, 297, 228, 289],
    #     "Foot": [263, 257, 258, 240, 231, 263, 242, 245, 255, 253]
    # }
    #
    # step2_max_data = {
    #     "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
    #     "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
    #     "Neck": [47, 32, 50, 47, 52, 47, 32, 55, 52, 49],
    #     "Elbow": [260, 359, 143, 164, 358, 260, 90, 156, 306, 359],
    #     "Bend": [10, 13, 14, 16, 27, 10, 14, 15, 12, 15],
    #     "Shoulder": [272, 262, 59, 69, 75, 272, 64, 69, 257, 267],
    #     "Torso": [228, 244, 250, 266, 242, 228, 242, 224, 229, 240],
    #     "Knee": [290, 274, 300, 304, 302, 290, 280, 297, 305, 303],
    #     "Foot": [259, 244, 236, 248, 261, 259, 245, 251, 259, 246]
    # }
    #
    # step3_max_data = {
    #     "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
    #     "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
    #     "Neck": [35, 27, 27, 30, 40, 35, 24, 26, 36, 33],
    #     "Elbow": [85, 88, 96, 111, 95, 85, 79, 187, 86, 158],
    #     "Bend": [8, 14, 10, 10, 14, 8, 14, 12, 9, 10],
    #     "Shoulder": [57, 61, 54, 60, 75, 57, 55, 62, 57, 60],
    #     "Torso": [183, 190, 187, 189, 220, 183, 188, 187, 186, 189],
    #     "Knee": [182, 197, 191, 207, 241, 182, 185, 190, 169, 184],
    #     "Foot": [184, 196, 213, 210, 255, 184, 192, 194, 200, 191]
    # }
    #
    # step4_max_data = {
    #     "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
    #     "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
    #     "Neck": [48, 30, 57, 48, 32, 48, 45, 36, 45, 69],
    #     "Elbow": [359, 90, 97, 123, 99, 359, 205, 102, 203, 33],
    #     "Bend": [30, 11, 18, 17, 15, 30, 14, 16, 25, 20],
    #     "Shoulder": [266, 63, 60, 65, 56, 266, 259, 57, 268, 270],
    #     "Torso": [233, 220, 213, 232, 187, 233, 219, 216, 222, 226],
    #     "Knee": [256, 249, 239, 272, 180, 256, 228, 228, 246, 246],
    #     "Foot": [268, 250, 265, 242, 202, 268, 252, 255, 257, 253]
    # }

    step1_min_data = {
        "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
        "Neck": [20, 21, 7, 10, 9, 20, 19, 18, 18, 22],
        "Elbow": [16, 0, 75, 83, 1, 16, 76, 74, 64, 69],
        "Bend": [6, 2, 2, 7, 1, 6, 8, 0, 3, 0],
        "Shoulder": [65, 63, 44, 41, 57, 65, 55, 56, 45, 48],
        "Torso": [180, 183, 188, 179, 167, 180, 187, 179, 188, 182],
        "Knee": [176, 193, 194, 179, 186, 176, 190, 177, 187, 183],
        "Foot": [182, 215, 200, 181, 176, 182, 201, 194, 206, 198]
    }

    step2_min_data = {
        "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
        "Neck": [13, 19, 4, 19, 7, 13, 19, 23, 16, 2],
        "Elbow": [6, 2, 74, 80, 12, 6, 72, 60, 19, 0],
        "Bend": [0, 3, 0, 1, 0, 0, 7, 0, 0, 0],
        "Shoulder": [1, 0, 15, 18, 0, 1, 50, 19, 0, 15],
        "Torso": [179, 179, 187, 176, 183, 179, 183, 171, 177, 164],
        "Knee": [174, 171, 175, 166, 181, 174, 177, 169, 172, 145],
        "Foot": [174, 170, 155, 179, 193, 174, 186, 166, 182, 165]
    }

    step3_min_data = {
        "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
        "Neck": [21, 22, 22, 22, 19, 21, 19, 17, 22, 26],
        "Elbow": [68, 81, 77, 87, 69, 68, 72, 75, 73, 72],
        "Bend": [4, 8, 4, 4, 0, 4, 9, 5, 4, 3],
        "Shoulder": [51, 56, 45, 49, 57, 51, 50, 17, 51, 32],
        "Torso": [176, 180, 180, 180, 188, 176, 182, 173, 178, 179],
        "Knee": [170, 173, 182, 174, 178, 170, 177, 171, 172, 169],
        "Foot": [161, 172, 194, 172, 211, 161, 177, 166, 171, 161]
    }

    step4_min_data = {
        "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
        "Neck": [11, 19, 10, 10, 18, 11, 21, 18, 8, 14],
        "Elbow": [83, 72, 67, 84, 79, 83, 178, 64, 156, 158],
        "Bend": [0, 4, 0, 0, 7, 0, 3, 3, 0, 0],
        "Shoulder": [80, 53, 37, 44, 53, 80, 78, 45, 71, 19],
        "Torso": [178, 182, 192, 176, 181, 178, 181, 188, 181, 184],
        "Knee": [175, 173, 189, 173, 172, 175, 177, 187, 175, 165],
        "Foot": [183, 211, 219, 183, 179, 183, 181, 206, 168, 149]
    }

    step1_max_data = {
        "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
        "Neck": [31, 44, 31, 32, 68, 31, 27, 44, 36, 41],
        "Elbow": [38, 359, 109, 115, 359, 38, 88, 90, 102, 90],
        "Bend": [14, 11, 22, 28, 12, 14, 15, 29, 16, 27],
        "Shoulder": [76, 265, 60, 68, 83, 76, 62, 55, 57, 292],
        "Torso": [279, 250, 254, 253, 268, 279, 259, 246, 216, 240],
        "Knee": [312, 299, 323, 282, 312, 312, 298, 297, 228, 289],
        "Foot": [263, 257, 258, 240, 231, 263, 242, 245, 255, 253]
    }

    step2_max_data = {
        "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
        "Neck": [47, 32, 50, 47, 52, 47, 32, 55, 52, 49],
        "Elbow": [260, 359, 143, 164, 358, 260, 90, 156, 306, 359],
        "Bend": [10, 13, 14, 16, 27, 10, 14, 15, 12, 15],
        "Shoulder": [272, 262, 59, 69, 75, 272, 64, 69, 257, 267],
        "Torso": [228, 244, 250, 266, 242, 228, 242, 224, 229, 240],
        "Knee": [290, 274, 300, 304, 302, 290, 280, 297, 305, 303],
        "Foot": [259, 244, 236, 248, 261, 259, 245, 251, 259, 246]
    }

    step3_max_data = {
        "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
        "Neck": [35, 27, 27, 30, 40, 35, 24, 26, 36, 33],
        "Elbow": [85, 88, 96, 111, 95, 85, 79, 187, 86, 158],
        "Bend": [8, 14, 10, 10, 14, 8, 14, 12, 9, 10],
        "Shoulder": [57, 61, 54, 60, 75, 57, 55, 62, 57, 60],
        "Torso": [183, 190, 187, 189, 220, 183, 188, 187, 186, 189],
        "Knee": [182, 197, 191, 207, 241, 182, 185, 190, 169, 184],
        "Foot": [184, 196, 213, 210, 255, 184, 192, 194, 200, 191]
    }

    step4_max_data = {
        "Category": [4, 1, 1, 2, 3, 4, 1, 1, 2, 3],
        "Day": [1, 1, 1, 1, 1, 40, 40, 40, 40, 40],
        "Neck": [48, 30, 57, 48, 32, 48, 45, 36, 45, 69],
        "Elbow": [359, 90, 97, 123, 99, 359, 205, 102, 203, 33],
        "Bend": [30, 11, 18, 17, 15, 30, 14, 16, 25, 20],
        "Shoulder": [266, 63, 60, 65, 56, 266, 259, 57, 268, 270],
        "Torso": [233, 220, 213, 232, 187, 233, 219, 216, 222, 226],
        "Knee": [256, 249, 239, 272, 180, 256, 228, 228, 246, 246],
        "Foot": [268, 250, 265, 242, 202, 268, 252, 255, 257, 253]
    }

    # Define the directory to save the plots
    save_dir = "/Users/vl/Documents/Kuchipudi_Thesis/plots"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # List of data dictionaries
    datasets = [step1_min_data, step2_min_data, step3_min_data, step4_min_data, step1_max_data, step2_max_data,
                step3_max_data, step4_max_data]

    # List of step names
    steps = ['Step1_min', 'Step2_min', 'Step3_min', 'Step4_min', 'Step1_max', 'Step2_max', 'Step3_max', 'Step4_max']

    # Iterate over each dataset and corresponding step name
    for step_name, data in zip(steps, datasets):
        # Create subfolder for each step
        step_dir = os.path.join(save_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)

        df = pd.DataFrame(data)

        # Filtering data for categories 1, 2, 3, and 4
        cat_1 = df[df['Category'] == 1]
        cat_2 = df[df['Category'] == 2]
        cat_3 = df[df['Category'] == 3]
        cat_4 = df[df['Category'] == 4]

        # Calculating means for each category and each day
        cat_1_means = cat_1.groupby('Day').mean().reset_index()
        cat_2_means = cat_2.groupby('Day').mean().reset_index()
        cat_3_means = cat_3.groupby('Day').mean().reset_index()
        cat_4_means = cat_4.groupby('Day').mean().reset_index()

        # Plotting comparison between categories 1, 2, 3, and 4 for each feature
        features = ['Neck', 'Elbow', 'Shoulder', 'Bend', 'Torso', 'Knee', 'Foot']
        plt.figure(figsize=(15, 10))  # Adjust the figure size as needed

        for i, feature in enumerate(features, start=1):
            plt.subplot(3, 3, i)
            plt.plot(cat_4_means['Day'], cat_4_means[feature], label='Category 4', marker='o')
            plt.plot(cat_1_means['Day'], cat_1_means[feature], label='Category 1', marker='o')
            plt.plot(cat_2_means['Day'], cat_2_means[feature], label='Category 2', marker='o')
            plt.plot(cat_3_means['Day'], cat_3_means[feature], label='Category 3', marker='o')

            plt.xlabel('Day')
            plt.ylabel('Mean Values')
            plt.title(f'Comparison of Categories 1, 2, 3 with Category 4 ({feature})')
            plt.legend()
            plt.grid(True)

        # Adjust layout to prevent overlap of subplots
        plt.tight_layout()

        # Save the plot inside the subfolder with the appropriate name
        plt.savefig(os.path.join(step_dir, 'All_Features.png'))
        plt.close()

