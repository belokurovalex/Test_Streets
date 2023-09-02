# Test_Streets
Road building programme
Algorithm of the programme:
1 Loading data from the road.shp file.
2. Copying coordinates of points to DataFrame Points_lines.
3. Creating auxiliary DataFrame Segments with a set of fields: 'x1', 'y1', 'x2', 'y2', 'index_point_1', 'index_point_2', 'selected'.
4. Write coordinates of line segments from Points_lines to Segments.
5. Calculating the indices for the transition of their each point of each segment. 
For this purpose, the number of points with the same coordinates is determined sequentially for each segment from the Segments table. 
  - If there is only one point, it is the beginning or the end of the line.
  - If there are two points, it is the continuation of the line (an intermediate point between two segments).
  - If there are three points, it is a T-junction. To decide which road is a dead end, the angles between the three vectors containing that point are calculated. 
    The vector with the largest angles is a dead end.
  - If there are four points, it is the intersection of two roads. To determine whether each of the four segments belongs to two roads, the angles between the           vectors are calculated. The segments with the smallest angles belong to one road.
Indices are written in the fields 'index_point_1', 'index_point_2'.
5. Display the roads on the plot. Each road is drawn in a different colour. The colours may be repeated. 
To draw, first, the elements marked in the fields 'index_point_1', 'index_point_2' as '-1' are found. 
The coordinates of each line segment are read from them, moving along the indices written in the 'index_point_1', 'index_point_2' fields. The read segments are marked as 'True' in the 'selected' field. The line construction is finished when the point '-1' is reached.
The ring road is built after all linear roads are completed. For this purpose, the remaining unselected elements are searched by analysing the 'selected' field.
The found elements are displayed on the graph.


