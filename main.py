#+++++++++++++++++++++++++++++++++++++ Program Streets++++++++++++++++++++++++++++++++

import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import math

#Calculating the angle between vectors
def angle_vectors(v1, v2):
    dot_pr = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(dot_pr / norms))


# Load data from roads.shp
data = gpd.read_file('D:/SMO/python-gis-test-task-master/sample/roads.shp', geometry=True)

#Get DataFrame points from input data
points_lines = data.get_coordinates()

#Create DataFrame segments
segments = pd.DataFrame()
#Append fields to DataFrame segments
segments['x1'] = 0
segments['y1'] = 0
segments['x2'] = 0
segments['y2'] = 0
segments['index_point_1'] = -2
segments['index_point_2'] = -2
segments['selected'] = False

#Size of points_lines
count_points = points_lines.shape

#Fill segments from points_lines
counter = 0
for number_line in range(0, count_points[0]-1):
    # Read segment from points_lines
    index_1 = points_lines.index[number_line]
    index_2 = points_lines.index[number_line+1]
    if (index_1 == index_2):
        w = points_lines.iloc[number_line]
        ww = w['x']
        segments.at[counter, 'x1'] = ww.astype('float64')
        w = points_lines.iloc[number_line]
        ww = w['y']
        segments.at[counter, 'y1'] = ww.astype('float64')
        w = points_lines.iloc[number_line+1]
        ww = w['x']
        segments.at[counter, 'x2'] = ww.astype('float64')
        w = points_lines.iloc[number_line+1]
        ww = w['y']
        segments.at[counter, 'y2'] = ww.astype('float64')

        segments.at[counter, 'index_point_1'] = segments.at[counter, 'index_point_2'] = -2
        segments.at[counter, 'selected'] = False
        counter = counter + 1

#Size of segments
count_segments = segments.shape

#Calculate the indexes to next points and write to fields 'index_point_1', 'index_point_2'
# ((-1) - first (last) point of line)

# +++++++++++++++++++++++First point of segments+++++++++++++++++++++++
for number_line in range(0, count_segments[0]):
    #Read one segment from DataFrame segments
    segments_line = segments.values[number_line]
    #Search Point
    search_point = segments_line[0]
    #Search for points with coincident coordinates in DataFrame segments
    search_result = segments.isin([search_point])
    # Filtering by fields: by 'x1' to indexes_1, by 'x2' to indexes_2
    indexes_1 = search_result.index[search_result['x1'] == True]
    indexes_2 = search_result.index[search_result['x2'] == True]
    #Count points with coincident coordinates in position point_1
    size_cross_1 = indexes_1.shape
    #Count points with coincident coordinates in position point_2
    size_cross_2 = indexes_2.shape
    sum_size_cross = size_cross_1[0] + size_cross_2[0]
    #One point
    if (sum_size_cross == 1):
        if (size_cross_1[0] == 1):
            segments.at[indexes_1[0], 'index_point_1'] = -1
        else:
            segments.at[indexes_2[0], 'index_point_2'] = -1
    #Two points
    if (sum_size_cross == 2):
        if (size_cross_1[0] == 1):
            segments.at[indexes_1[0], 'index_point_1'] = indexes_2[0]
            segments.at[indexes_2[0], 'index_point_2'] = indexes_1[0]
        else:
            segments.at[indexes_1[0], 'index_point_1'] = indexes_1[1]
            segments.at[indexes_1[1], 'index_point_1'] = indexes_1[0]
    #Three points
    if (sum_size_cross == 3):
        flag = 0
        # Points from indexes_1
        #Fill vectors: vector_1, vector_2, vector_3
        for number_points_1 in range(0, size_cross_1[0]):
            if (flag == 0):
                w1 = segments.iloc[indexes_1[number_points_1]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_1 = np.array([n_x, n_y])
                vector_1_number_line = indexes_1[number_points_1]
                vector_1_point_left_right = 'index_point_1'
            if (flag == 1):
                w1 = segments.iloc[indexes_1[number_points_1]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_2 = np.array([n_x, n_y])
                vector_2_number_line = indexes_1[number_points_1]
                vector_2_point_left_right = 'index_point_1'
            if (flag == 2):
                w1 = segments.iloc[indexes_1[number_points_1]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_3 = np.array([n_x, n_y])
                vector_3_number_line = indexes_1[number_points_1]
                vector_3_point_left_right = 'index_point_1'

            flag = flag + 1

        # Points from indexes_2
        #Fill vectors: vector_1, vector_2, vector_3
        for number_points_2 in range(0, size_cross_2[0]):
            if (flag == 0):
                w1 = segments.iloc[indexes_2[number_points_2]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_1 = np.array([n_x, n_y])
                vector_1_number_line = indexes_2[number_points_2]
                vector_1_point_left_right = 'index_point_2'
            if (flag == 1):
                w1 = segments.iloc[indexes_2[number_points_2]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_2 = np.array([n_x, n_y])
                vector_2_number_line = indexes_2[number_points_2]
                vector_2_point_left_right = 'index_point_2'
            if (flag == 2):
                w1 = segments.iloc[indexes_2[number_points_2]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_3 = np.array([n_x, n_y])
                vector_3_number_line = indexes_2[number_points_2]
                vector_3_point_left_right = 'index_point_2'

            flag = flag + 1

        #Calculate angles
        angle_1_2 = angle_vectors(vector_1, vector_2)
        angle_1_3 = angle_vectors(vector_1, vector_3)
        angle_2_3 = angle_vectors(vector_2, vector_3)

        #Record the indexes to next points
        if angle_1_3 >= angle_1_2 <= angle_2_3:
            segments.at[vector_1_number_line, vector_1_point_left_right] = vector_2_number_line
            segments.at[vector_2_number_line, vector_2_point_left_right] = vector_1_number_line
            segments.at[vector_3_number_line, vector_3_point_left_right] = -1
        elif angle_1_2 >= angle_1_3 <= angle_2_3:
            segments.at[vector_1_number_line, vector_1_point_left_right] = vector_3_number_line
            segments.at[vector_3_number_line, vector_3_point_left_right] = vector_1_number_line
            segments.at[vector_2_number_line, vector_2_point_left_right] = -1
        else:
            segments.at[vector_2_number_line, vector_2_point_left_right] = vector_3_number_line
            segments.at[vector_3_number_line, vector_3_point_left_right] = vector_2_number_line
            segments.at[vector_1_number_line, vector_1_point_left_right] = -1

    #Four points
    if (sum_size_cross == 4):
        flag = 0
        # Points from indexes_1
        #Fill vectors: vector_1, vector_2, vector_3, vector_4
        for number_points_1 in range(0, size_cross_1[0]):
            if (flag == 0):
                w1 = segments.iloc[indexes_1[number_points_1]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_1 = np.array([n_x, n_y])
                vector_1_number_line = indexes_1[number_points_1]
                vector_1_point_left_right = 'index_point_1'
            if (flag == 1):
                w1 = segments.iloc[indexes_1[number_points_1]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_2 = np.array([n_x, n_y])
                vector_2_number_line = indexes_1[number_points_1]
                vector_2_point_left_right = 'index_point_1'
            if (flag == 2):
                w1 = segments.iloc[indexes_1[number_points_1]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_3 = np.array([n_x, n_y])
                vector_3_number_line = indexes_1[number_points_1]
                vector_3_point_left_right = 'index_point_1'
            if (flag == 3):
                w1 = segments.iloc[indexes_1[number_points_1]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_4 = np.array([n_x, n_y])
                vector_4_number_line = indexes_1[number_points_1]
                vector_4_point_left_right = 'index_point_1'

            flag = flag + 1

        # Points from indexes_2
        #Fill vectors: vector_1, vector_2, vector_3, vector_4
        for number_points_2 in range(0, size_cross_2[0]):
            if (flag == 0):
                w1 = segments.iloc[indexes_2[number_points_2]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_1 = np.array([n_x, n_y])
                vector_1_number_line = indexes_2[number_points_2]
                vector_1_point_left_right = 'index_point_2'
            if (flag == 1):
                w1 = segments.iloc[indexes_2[number_points_2]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_2 = np.array([n_x, n_y])
                vector_2_number_line = indexes_2[number_points_2]
                vector_2_point_left_right = 'index_point_2'
            if (flag == 2):
                w1 = segments.iloc[indexes_2[number_points_2]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_3 = np.array([n_x, n_y])
                vector_3_number_line = indexes_2[number_points_2]
                vector_3_point_left_right = 'index_point_2'
            if (flag == 3):
                w1 = segments.iloc[indexes_2[number_points_2]]
                n_x = w1['x2'] - w1['x1']
                n_y = w1['y2'] - w1['y1']
                vector_4 = np.array([n_x, n_y])
                vector_4_number_line = indexes_2[number_points_2]
                vector_4_point_left_right = 'index_point_2'

            flag = flag + 1

        #Calculate angles
        angle_1_2 = angle_vectors(vector_1, vector_2)
        angle_1_3 = angle_vectors(vector_1, vector_3)
        angle_1_4 = angle_vectors(vector_1, vector_4)
        angle_2_3 = angle_vectors(vector_2, vector_3)
        angle_2_4 = angle_vectors(vector_2, vector_4)
        angle_3_4 = angle_vectors(vector_3, vector_4)

        #Record the indexes to next points
        if angle_1_3 >= angle_1_2 <= angle_1_4:
            segments.at[vector_1_number_line, vector_1_point_left_right] = vector_2_number_line
            segments.at[vector_2_number_line, vector_2_point_left_right] = vector_1_number_line
            segments.at[vector_3_number_line, vector_3_point_left_right] = vector_4_number_line
            segments.at[vector_4_number_line, vector_4_point_left_right] = vector_3_number_line
        elif angle_1_2 >= angle_1_3 <= angle_1_4:
            segments.at[vector_1_number_line, vector_1_point_left_right] = vector_3_number_line
            segments.at[vector_3_number_line, vector_3_point_left_right] = vector_1_number_line
            segments.at[vector_2_number_line, vector_2_point_left_right] = vector_4_number_line
            segments.at[vector_4_number_line, vector_4_point_left_right] = vector_2_number_line
        else:
            segments.at[vector_1_number_line, vector_1_point_left_right] = vector_4_number_line
            segments.at[vector_4_number_line, vector_4_point_left_right] = vector_1_number_line
            segments.at[vector_2_number_line, vector_2_point_left_right] = vector_3_number_line
            segments.at[vector_3_number_line, vector_3_point_left_right] = vector_2_number_line
#---------------------------First point of segments--------------------------

# +++++++++++++++++++++++Second point of segments+++++++++++++++++++++++
for number_line in range(0, count_segments[0]):
    #Read one segment from DataFrame segments
    segments_line = segments.values[number_line]
    #Taste field
    if (segments_line[5] == -2):
        # Search Point
        search_point = segments_line[2]
    #Search for points with coincident coordinates in DataFrame segments
        kk = segments.isin([search_point])
    # Filtering by fields: by 'x1' to indexes_1, by 'x2' to indexes_2
        aa = kk[kk == True]
        indexes_1 = kk.index[kk['x1'] == True]
        indexes_2 = kk.index[kk['x2'] == True]
    #Count points with coincident coordinates in position point_1
        size_cross_1 = indexes_1.shape
    #Count points with coincident coordinates in position point_2
        size_cross_2 = indexes_2.shape
        sum_size_cross = size_cross_1[0] + size_cross_2[0]

        # One point
        if (sum_size_cross == 1):
            if (size_cross_2[0] == 1):
                segments.at[indexes_2[0], 'index_point_2'] = -1
            else:
                segments.at[indexes_1[0], 'index_point_1'] = -1

        # Two points
        if (sum_size_cross == 2):
            if (size_cross_2[0] == 1):
                segments.at[indexes_2[0], 'index_point_2'] = indexes_1[0]
                segments.at[indexes_1[0], 'index_point_1'] = indexes_2[0]
            else:
                segments.at[indexes_2[0], 'index_point_2'] = indexes_2[1]
                segments.at[indexes_2[1], 'index_point_2'] = indexes_2[0]

        # Three points
        if (sum_size_cross == 3):
            flag = 0
            # Points from indexes_2
            # Fill vectors: vector_1, vector_2, vector_3
            for number_points_2 in range(0, size_cross_2[0]):
                if (flag == 0):
                    w1 = segments.iloc[indexes_2[number_points_2]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_1 = np.array([n_x, n_y])
                    vector_1_number_line = indexes_2[number_points_2]
                    vector_1_point_left_right = 'index_point_2'
                if (flag == 1):
                    w1 = segments.iloc[indexes_2[number_points_2]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_2 = np.array([n_x, n_y])
                    vector_2_number_line = indexes_2[number_points_2]
                    vector_2_point_left_right = 'index_point_2'
                if (flag == 2):
                    w1 = segments.iloc[indexes_2[number_points_2]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_3 = np.array([n_x, n_y])
                    vector_3_number_line = indexes_2[number_points_2]
                    vector_3_point_left_right = 'index_point_2'

                flag = flag + 1

            # Points from indexes_1
            # Fill vectors: vector_1, vector_2, vector_3
            for number_points_1 in range(0, size_cross_1[0]):
                if (flag == 0):
                    w1 = segments.iloc[indexes_1[number_points_1]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_1 = np.array([n_x, n_y])
                    vector_1_number_line = indexes_1[number_points_1]
                    vector_1_point_left_right = 'index_point_1'
                if (flag == 1):
                    w1 = segments.iloc[indexes_1[number_points_1]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_2 = np.array([n_x, n_y])
                    vector_2_number_line = indexes_1[number_points_1]
                    vector_2_point_left_right = 'index_point_1'
                if (flag == 2):
                    w1 = segments.iloc[indexes_1[number_points_1]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_3 = np.array([n_x, n_y])
                    vector_3_number_line = indexes_1[number_points_1]
                    vector_3_point_left_right = 'index_point_1'

                flag = flag + 1

            # Calculate angles
            angle_1_2 = angle_vectors(vector_1, vector_2)
            angle_1_3 = angle_vectors(vector_1, vector_3)
            angle_2_3 = angle_vectors(vector_2, vector_3)

            # Record the indexes to next points
            if angle_1_3 >= angle_1_2 <= angle_2_3:
                segments.at[vector_1_number_line, vector_1_point_left_right] = vector_2_number_line
                segments.at[vector_2_number_line, vector_2_point_left_right] = vector_1_number_line
                segments.at[vector_3_number_line, vector_3_point_left_right] = -1
            elif angle_1_2 >= angle_1_3 <= angle_2_3:
                segments.at[vector_1_number_line, vector_1_point_left_right] = vector_3_number_line
                segments.at[vector_3_number_line, vector_3_point_left_right] = vector_1_number_line
                segments.at[vector_2_number_line, vector_2_point_left_right] = -1
            else:
                segments.at[vector_2_number_line, vector_2_point_left_right] = vector_3_number_line
                segments.at[vector_3_number_line, vector_3_point_left_right] = vector_2_number_line
                segments.at[vector_1_number_line, vector_1_point_left_right] = -1

        # Four points
        if (sum_size_cross == 4):
            flag = 0
            # Points from indexes_2
            # Fill vectors: vector_1, vector_2, vector_3, vector_4
            for number_points_2 in range(0, size_cross_2[0]):
                if (flag == 0):
                    w1 = segments.iloc[indexes_2[number_points_2]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_1 = np.array([n_x, n_y])
                    vector_1_number_line = indexes_2[number_points_2]
                    vector_1_point_left_right = 'index_point_2'
                if (flag == 1):
                    w1 = segments.iloc[indexes_2[number_points_2]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_2 = np.array([n_x, n_y])
                    vector_2_number_line = indexes_2[number_points_2]
                    vector_2_point_left_right = 'index_point_2'
                if (flag == 2):
                    w1 = segments.iloc[indexes_2[number_points_2]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_3 = np.array([n_x, n_y])
                    vector_3_number_line = indexes_2[number_points_2]
                    vector_3_point_left_right = 'index_point_2'
                if (flag == 3):
                    w1 = segments.iloc[indexes_2[number_points_2]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_4 = np.array([n_x, n_y])
                    vector_4_number_line = indexes_2[number_points_2]
                    vector_4_point_left_right = 'index_point_2'

                flag = flag + 1

            # Points from indexes_1
            # Fill vectors: vector_1, vector_2, vector_3, vector_4
            for number_points_1 in range(0, size_cross_1[0]):
                if (flag == 0):
                    w1 = segments.iloc[indexes_1[number_points_1]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    # n_x = -n_x
                    # n_y = -n_y
                    vector_1 = np.array([n_x, n_y])
                    vector_1_number_line = indexes_1[number_points_1]
                    vector_1_point_left_right = 'index_point_1'
                if (flag == 1):
                    w1 = segments.iloc[indexes_1[number_points_1]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_2 = np.array([n_x, n_y])
                    vector_2_number_line = indexes_1[number_points_1]
                    vector_2_point_left_right = 'index_point_1'
                if (flag == 2):
                    w1 = segments.iloc[indexes_1[number_points_1]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_3 = np.array([n_x, n_y])
                    vector_3_number_line = indexes_1[number_points_1]
                    vector_3_point_left_right = 'index_point_1'
                if (flag == 3):
                    w1 = segments.iloc[indexes_1[number_points_1]]
                    n_x = w1['x2'] - w1['x1']
                    n_y = w1['y2'] - w1['y1']
                    vector_4 = np.array([n_x, n_y])
                    vector_4_number_line = indexes_1[number_points_1]
                    vector_4_point_left_right = 'index_point_1'

                flag = flag + 1

            # Calculate angles
            angle_1_2 = angle_vectors(vector_1, vector_2)
            angle_1_3 = angle_vectors(vector_1, vector_3)
            angle_1_4 = angle_vectors(vector_1, vector_4)
            angle_2_3 = angle_vectors(vector_2, vector_3)
            angle_2_4 = angle_vectors(vector_2, vector_4)
            angle_3_4 = angle_vectors(vector_3, vector_4)

            # Record the indexes to next points
            if angle_1_3 >= angle_1_2 <= angle_1_4:
                segments.at[vector_1_number_line, vector_1_point_left_right] = vector_2_number_line
                segments.at[vector_2_number_line, vector_2_point_left_right] = vector_1_number_line
                segments.at[vector_3_number_line, vector_3_point_left_right] = vector_4_number_line
                segments.at[vector_4_number_line, vector_4_point_left_right] = vector_3_number_line
            elif angle_1_2 >= angle_1_3 <= angle_1_4:
                segments.at[vector_1_number_line, vector_1_point_left_right] = vector_3_number_line
                segments.at[vector_3_number_line, vector_3_point_left_right] = vector_1_number_line
                segments.at[vector_2_number_line, vector_2_point_left_right] = vector_4_number_line
                segments.at[vector_4_number_line, vector_4_point_left_right] = vector_2_number_line
            else:
                segments.at[vector_1_number_line, vector_1_point_left_right] = vector_4_number_line
                segments.at[vector_4_number_line, vector_4_point_left_right] = vector_1_number_line
                segments.at[vector_2_number_line, vector_2_point_left_right] = vector_3_number_line
                segments.at[vector_3_number_line, vector_3_point_left_right] = vector_2_number_line
#-----------------------------Second point of segments--------------------------------

#+++++++++++++++++++++++++++++++++Lines to plot+++++++++++++++++++++++++++++++++++++++

# plot the data
fig, ax = plt.subplots(figsize=(30, 30))

#Searching for a tail
kk = segments.isin([-1])
indexes_1 = kk.index[kk['index_point_1'] == True]
indexes_2 = kk.index[kk['index_point_2'] == True]
# Count points in field 'index_point_1'
size_cross_1 = indexes_1.shape
# Count points in field 'index_point_2'
size_cross_2 = indexes_2.shape

# +++++++++++++++++++++++++Points for field 'index_point_1'+++++++++++++++++++++++++
for number_points_1 in range(0, size_cross_1[0]):
    w1 = segments.iloc[indexes_1[number_points_1]]
    num_index_point_left = w1['index_point_1']
    num_index_point_right = w1['index_point_2']
    selected_segment = w1['selected']
    x_y = pd.DataFrame()
    #Append fields to DataFrame segments
    x_y['x'] = 0
    x_y['y'] = 0
    index_x = 0
    if (selected_segment == False):
        x_y.at[index_x, 'x'] = w1['x1']
        x_y.at[index_x, 'y'] = w1['y1']
        index_x = index_x + 1
        x_y.at[index_x, 'x'] = w1['x2']
        x_y.at[index_x, 'y'] = w1['y2']
        index_x = index_x + 1
        segments.at[indexes_1[number_points_1], 'selected'] = True

    number_string = indexes_1[number_points_1]
    if (num_index_point_left == -1):
        number_string_jump = num_index_point_right.astype('int64')
    else:
        number_string_jump = num_index_point_left.astype('int64')

    while (number_string_jump != -1):
        w1 = segments.iloc[number_string_jump]
        num_index_point_left = w1['index_point_1']
        num_index_point_right = w1['index_point_2']
        selected_segment = w1['selected']

        if (num_index_point_left == number_string):
            if (selected_segment == False):
                x_y.at[index_x, 'x'] = w1['x1']
                x_y.at[index_x, 'y'] = w1['y1']
                index_x = index_x + 1
                x_y.at[index_x, 'x'] = w1['x2']
                x_y.at[index_x, 'y'] = w1['y2']
                index_x = index_x + 1
                segments.at[number_string_jump, 'selected'] = True

            number_string_jump_prev = number_string_jump
            number_string_jump = num_index_point_right.astype('int64')
        else:
            if (selected_segment == False):
                x_y.at[index_x, 'x'] = w1['x2']
                x_y.at[index_x, 'y'] = w1['y2']
                index_x = index_x + 1
                x_y.at[index_x, 'x'] = w1['x1']
                x_y.at[index_x, 'y'] = w1['y1']
                index_x = index_x + 1
                segments.at[number_string_jump, 'selected'] = True

            number_string_jump_prev = number_string_jump
            number_string_jump = num_index_point_left.astype('int64')

        number_string = number_string_jump_prev

    x = x_y["x"]
    y = x_y["y"]
    plt.plot(x, y)

#+++++++++++++++++++++++++++++Points for field 'index_point_2'++++++++++++++++++++++++++
for number_points_1 in range(0, size_cross_2[0]):
    w1 = segments.iloc[indexes_2[number_points_1]]
    num_index_point_left = w1['index_point_1']
    num_index_point_right = w1['index_point_2']
    selected_segment = w1['selected']
    x_y = pd.DataFrame()
    #Append fields to DataFrame segments
    x_y['x'] = 0
    x_y['y'] = 0
    index_x = 0
    if (selected_segment == False):
        x_y.at[index_x, 'x'] = w1['x1']
        x_y.at[index_x, 'y'] = w1['y1']
        index_x = index_x + 1
        x_y.at[index_x, 'x'] = w1['x2']
        x_y.at[index_x, 'y'] = w1['y2']
        index_x = index_x + 1
        segments.at[indexes_2[number_points_1], 'selected'] = True

    number_string = indexes_2[number_points_1]
    if (num_index_point_left == -1):
        number_string_jump = num_index_point_right.astype('int64')
    else:
        number_string_jump = num_index_point_left.astype('int64')

    while (number_string_jump != -1):
        w1 = segments.iloc[number_string_jump]
        num_index_point_left = w1['index_point_1']
        num_index_point_right = w1['index_point_2']
        selected_segment = w1['selected']

        if (num_index_point_left == number_string):
            if (selected_segment == False):
                x_y.at[index_x, 'x'] = w1['x1']
                x_y.at[index_x, 'y'] = w1['y1']
                index_x = index_x + 1
                x_y.at[index_x, 'x'] = w1['x2']
                x_y.at[index_x, 'y'] = w1['y2']
                index_x = index_x + 1
                segments.at[number_string_jump, 'selected'] = True

            number_string_jump_prev = number_string_jump
            number_string_jump = num_index_point_right.astype('int64')
        else:
            if (selected_segment == False):
                x_y.at[index_x, 'x'] = w1['x2']
                x_y.at[index_x, 'y'] = w1['y2']
                index_x = index_x + 1
                x_y.at[index_x, 'x'] = w1['x1']
                x_y.at[index_x, 'y'] = w1['y1']
                index_x = index_x + 1
                segments.at[number_string_jump, 'selected'] = True

            number_string_jump_prev = number_string_jump
            number_string_jump = num_index_point_left.astype('int64')

        number_string = number_string_jump_prev

    x = x_y["x"]
    y = x_y["y"]
    col = (np.random.random(), np.random.random(), np.random.random())
    plt.plot(x, y, c=col)

#Searching for circles
kk = segments.isin([False])
indexes_1 = kk.index[kk['selected'] == True]
# Count points in field 'selected'
size_cross_1 = indexes_1.shape

# +++++++++++++++++++++++++Points for field 'selected'+++++++++++++++++++++++++
for number_points_1 in range(0, size_cross_1[0]):
    w1 = segments.iloc[indexes_1[number_points_1]]
    num_index_point_left = w1['index_point_1']
    num_index_point_right = w1['index_point_2']
    selected_segment = w1['selected']
    num_index_point_left_stop = num_index_point_left
    x_y = pd.DataFrame()
    #Append fields to DataFrame segments
    x_y['x'] = 0
    x_y['y'] = 0
    index_x = 0

    number_string_jump = num_index_point_left.astype('int64')

    count_elements = 0
    while (number_string_jump != -1):
        w1 = segments.iloc[number_string_jump]
        num_index_point_left = w1['index_point_1']
        num_index_point_right = w1['index_point_2']
        selected_segment = w1['selected']

        if (num_index_point_left == number_string):
            if (selected_segment == False):
                x_y.at[index_x, 'x'] = w1['x1']
                x_y.at[index_x, 'y'] = w1['y1']
                index_x = index_x + 1
                x_y.at[index_x, 'x'] = w1['x2']
                x_y.at[index_x, 'y'] = w1['y2']
                index_x = index_x + 1
                segments.at[number_string_jump, 'selected'] = True
                count_elements = count_elements +1

            number_string_jump_prev = number_string_jump
            number_string_jump = num_index_point_right.astype('int64')

        else:
            if (selected_segment == False):
                x_y.at[index_x, 'x'] = w1['x2']
                x_y.at[index_x, 'y'] = w1['y2']
                index_x = index_x + 1
                x_y.at[index_x, 'x'] = w1['x1']
                x_y.at[index_x, 'y'] = w1['y1']
                index_x = index_x + 1
                segments.at[number_string_jump, 'selected'] = True
                count_elements = count_elements +1

            number_string_jump_prev = number_string_jump
            number_string_jump = num_index_point_left.astype('int64')

        if (num_index_point_left == num_index_point_left_stop):
            number_string_jump = -1

        number_string = number_string_jump_prev

    x = x_y["x"]
    y = x_y["y"]
    col = (np.random.random(), np.random.random(), np.random.random())
    if (count_elements > 2):
        plt.plot(x, y, c=col)

plt.show()


