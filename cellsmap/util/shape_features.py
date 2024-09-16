import numpy as np
from skimage import filters
from skimage import measure
from skimage import draw
from skimage import morphology



def arr2graph(arr: np.array) -> np.array:
    """Will take a binary image array showing a network-like structure
    and return the labeled versions of the nodes, edges, skeletons and
    pixel connectivity in that order. The connectivity equal to the
    dimensionality of arr (if 2D then a connectivity of 2 or a 3x3 square
    is used, if 3D then a connectivity of 3 or a 3x3x3 cube is used).
    
    Parameters
    ----------
    arr: np.array
        A binary 2D or 3D numpy array representing an image with dendritic, branching,
        tree-like, or network-like structures.

    Returns
    -------
    nodes_lab: np.array
        The nodes in arr where each node has a unique label as an array of the same
        shape as arr.

    edges_lab: np.array
        The edges in arr where each edge has a unique label as an array of the same
        shape as arr.

    skels_lab: np.array
        The skeletonization of arr where unconnected skeletons have unique labels as
        an array of the same shape as arr.

    conn: np.array
        The connectivity of each pixel in arr as an array of the same shape as arr.
    """

    ## Make sure that the array is either 2D or 3D
    assert(arr.ndim == 2 or arr.ndim == 3), 'Input array must be 2D or 3D.'

    if arr.ndim == 2:
        footprint = morphology.square(3)
    elif arr.ndim == 3:
        footprint = morphology.cube(3)

    ## Fill any tiny holes
    arr_filled = morphology.binary_closing(arr, footprint=footprint)
    skel = morphology.skeletonize(arr_filled).astype(bool)
    ## skeletonize above will make your array int8 dtype, and
    ## will make True == 255, but I want it to be 1, so I will
    ## force it to be bool, hence the .astype above.

    ## Converting the bool to int now does not make 
    ## True -> 255, instead True -> 1 (which is what I want):
    ## Transform the skeletonized array into one where each
    ## pixel has a value equal to the number of non-zero 
    ## immediate neighbors plus itself
    ## the * skel is to re-skeletonize the rank sum
    conn = filters.rank.pop(skel.astype(np.uint8), 
                            footprint=footprint,
                            mask=skel) * skel
    # This produces an array with the following values
    # (which is why I insisted on having the skeletonized array
    # have only 0s and 1s as values):
    # conn == 1,2 -> node (isolated point)
    # conn == 2 -> node (end point)
    # conn == 3 -> edge
    # conn >= 4 -> node (branch point)

    ## Label those endpoints, edges, and branchpoints (this is
    ## to get the connections between edges and nodes later on):
    edges_arr = (conn == 3)
    nodes_arr = ((conn == 1) + (conn == 2) + (conn >= 4))

    ## There can be both isolated nodes (a single pixel in space)
    ## and isolated edges (a closed loop in space)
    ## how do you uniquely define such a graph?
    ## Both edges and nodes need their own labels.
    nodes_lab = morphology.label(nodes_arr, connectivity=arr.ndim)
    edges_lab = morphology.label(edges_arr, connectivity=arr.ndim)
    skels_lab = morphology.label(skel, connectivity=arr.ndim)

    return nodes_lab, edges_lab, skels_lab, conn


def get_neighboring_labels(home_img: np.array, labeled_neighbors_img: np.array, bad_neighbors: list=None) -> tuple:
    """
    home_img will be made binary (can be an image where only a particular label was
    chosen by home_img == lab)
    bad_neighbors argument lets you choose labels in labeled_neighbors_img to exclude
    from result (e.g. 0 is often background, so may want to exclude 0).
    
    Parameters
    ----------
    home_img: np.array
        A binary array of the region you want to get the neighboring regions of.

    labeled_neighbors_img: np.array
        A labeled image of the same shape as home_img. Any labeled regions that are
        next to those found in home_img (as defined by a 3x3 square or 3x3x3 cube
        neighborhood for 2D or 3D, respectively) are considered a neighboring labels.

    bad_neighbors: list
        A list of region labels you want to ignore (0 is commonly reserved for the background).
        Default is None (i.e. do not ignore any values).

    Returns
    -------
    neighbors: tuple
        A tuple of the labels that are neighbors to home_img.
    """

    if home_img.ndim == 2:
        footprint = morphology.square(3)
    elif home_img.ndim == 3:
        footprint = morphology.cube(3)
    neighbors = [*np.unique(morphology.binary_dilation(home_img, footprint=footprint) * labeled_neighbors_img)]
    if bad_neighbors:
        # neighbors = [tuple([n for n in ns if n not in np.unique(bad_neighbors)]) for ns in neighbors]
        neighbors = [n for n in neighbors if n not in np.unique(bad_neighbors)]

    return tuple(neighbors)


def expand_bbox(bbox: tuple, ndim: int) -> tuple:
    """
    Take a bbox from skimages measure.regionprops and expands it by 1 pixel all around.
    Used to see 1 pixel away from a node or edge.

    Parameters
    ----------
    bbox: tuple
        This is a tuple of the form (row_start, col_start, row_end, col_end).

    ndim: int
        Number of dimensions in the image.

    Returns
    -------
    big_bbox: tuple
        The bbox expanded by 1 pixel. Has the same form as bbox.
    """

    big_bbox = (tuple((np.array(bbox[0:ndim]) - 0.5).astype(int)), tuple(np.array(bbox[ndim:2*ndim]) + 1))

    return big_bbox


def get_windows(img_lab: np.array) -> zip: #labeled_img
    """
    Takes a labeled image in the form of a numpy array and returns a zip of
    (labels, windows), where "labels" are labels in the labeled image and
    "windows" are lists of slice objects that define a bounding box.

    Parameters
    ----------
    img_lab: numpy array
        The labeled image as a numpy array.

    Returns
    -------
    lab_windows: zip
        A zip of the labels and windows.
    """

    img_lab_props = measure.regionprops(img_lab)
    ndim = img_lab.ndim

    # Create a list of labels and their associated bounding box:
    lab_labs, lab_bbox = zip(*[(lab.label, lab.bbox) for lab in img_lab_props])

    # Apparently Python now allows your upper slice range to exceed bounds, and instead
    # will just return the values within range.
    # Grab a bbox that is 1 pixel wider on each edge of each axis:
    lab_bbox_big = [expand_bbox(bbox, ndim) for bbox in lab_bbox]   

    # Create slicing windows of these expanded bboxes:
    windows = [[slice(*i) for i in list(zip(*bb))] for bb in lab_bbox_big]

    # zip the labels and windows together
    lab_windows = zip(lab_labs, windows)

    return lab_windows


def get_neighbor_nodes_and_edges(nodes_lab: np.array, edges_lab: np.array, bad_neighbors: list=[0], as_dict: bool=False):
    """
    Takes a labeled array of nodes and a labeled array of edges and returns
    a list or dict of which nodes neighbor each node, which edges neighbor each node,
    and which nodes neighbor each edge.
    The reason both lists of nodes and lists of edges are returned is because it is
    possible for a node or edge to have no neighbors. 
    This function is designed to work with the output of the arr2graph function.

    Parameters
    ----------
    nodes_lab: np.array
        The labeled nodes.

    edges_lab: np.array
        The labeled edges. Shape must be the same as nodes_lab.

    bad_neighbors: list
        A list of region labels to exclude when determining neighbors.
        Background labels are often 0.
        Default is [0].

    as_dict: bool
        If True returns the neighbors as a dictionary, otherwise a list.
        Default is False.

    Returns
    -------
    node_neighbors_edgelabs: list or dict
        Node labels coupled to their neighboring edge labels in the form
            [(node_label_1, (edge_label_1.1, edge_label_1.2, ...),
              node_label_2, (edge_label_2.1, edge_label_2.2, ...),
              ...)]

    edge_neighbors_nodelabs: list or dict
        Edge labels coupled to their neighboring node labels in the form
            [(edge_label_1, (node_label_1.1, node_label_1.2),
              edge_label_2, (node_label_2.1, node_label_2.2),
              ...)]

    node_neighbors_nodelabs: list or dict
        Node labels coupled to their neighboring node labels in the form
            [(node_label_1, (node_label_1.1, node_label_1.2, ...),
              node_label_2, (node_label_2.1, node_label_2.2, ...),
              ...)]
    """

    nodes_lab_windows = get_windows(nodes_lab)
    edges_lab_windows = get_windows(edges_lab)

    ## Find connected networks in the skeleton by labeling skel
    # skel_lab = morphology.label(skel, connectivity=3)

    ## Find which nodes neighbor which edges, and which edges neighbor which nodes:
    ## NOTE
    ## we are ensuring that only one node shows up when querying for neighbors in a
    ## window by setting nodes_lab == l (i.e. only show pixels that equal the label
    ## associated with the window)
    node_neighbors_edgelabs = [(l, get_neighboring_labels(nodes_lab[(*w,)]==l, edges_lab[(*w,)], bad_neighbors=bad_neighbors)) for l,w in nodes_lab_windows]
    edge_neighbors_nodelabs = [(l, get_neighboring_labels(edges_lab[(*w,)]==l, nodes_lab[(*w,)], bad_neighbors=bad_neighbors)) for l,w in edges_lab_windows]


    ## Use the combination of node_neighbors_edgelabs and edge_neighbors_nodelabs to
    ## find which nodes neighbor each other:
    # nodes_lab_props = measure.regionprops(nodes_lab, intensity_image=skel_lab)

    node_neighbors_nodelabs = []
    for x in node_neighbors_edgelabs:
        # Get which edges are connected to a particular node:
        node, edges = x
        # Iterate through the edge_neighbors and look for connected nodes
        # in the edge_neighbors_nodelabs list:
        node_neighbors_nodelabs.append((node, [n for e,n in edge_neighbors_nodelabs if e in edges]))
    # Clean up the node list with node neighbors so that  there are no repeating node labels
    node_neighbors_nodelabs = [(node, tuple(np.unique([n for ns in n_neighbors for n in ns]))) for node, n_neighbors in node_neighbors_nodelabs]
    # and also remove the "home node" from the node neighbors list to get the final cleaned up list:
    node_neighbors_nodelabs = [(node, tuple([n for n in n_neighbors if n != node])) for node, n_neighbors in node_neighbors_nodelabs]

    if as_dict:
        node_neighbors_edgelabs = {key:val for key,val in node_neighbors_edgelabs}
        edge_neighbors_nodelabs = {key:val for key,val in edge_neighbors_nodelabs}
        node_neighbors_nodelabs = {key:val for key,val in node_neighbors_nodelabs}

    return node_neighbors_edgelabs, edge_neighbors_nodelabs, node_neighbors_nodelabs


def numpy_mesh_coords(coord1_ls: list, coord2_ls: list, indexing: str='ij', return_indiv_coord_meshes: bool=False) -> list:
    """
    Performs a numpy meshgrid operation for coordinate points.

    Coordinate lists are lists of tuples, e.g.
    [(z1, y1, x1), (z2, y2, x2), ...]
    
    Parameters
    ----------
    coord1_ls: list of tuples
        Coordinate points to mesh with coord2_ls.
        e.g. [(z1, y1, x1), (z2, y2, x2), ...]

    coord2_ls: list of tuples 
        Coordinate points to mesh with coord1_ls.

    indexing: can be 'ij' or 'xy'.
        This is passed to the 'indexing' argument of numpy.meshgrid. Default is 'ij'.

    return_indiv_coord_meshes: bool
        If True will return a list of D numpy arrays, where D is the dimensionality
        of the coordinate points (e.g. meshing 2 lists of 2D (x,y) coordinates each of
        length 20 together will return a list of two 20x20 numpy arrays,
        i.e. list(x_coordinate_array, y_coordinate_array)).
        Otherwise, if False, the list of arrays will be stacked together into a single array
        of shape 20x20xD (e.g. in the previous example the 2 20x20 arrays would become a
        single 20x20x2 array).
        Default is False.

    Returns
    -------
    coord_meshes: list of numpy arrays
        List of numpy arrays representing the meshed coordinates.
    """

    coords1 = zip(*coord1_ls)
    coords2 = zip(*coord2_ls)

    coords = list(zip(coords1, coords2))
    coord_meshes = [np.meshgrid(*coord_ax, indexing=indexing) for coord_ax in coords]

    if not return_indiv_coord_meshes:
        coord_meshes = [np.dstack(coords) for coords in zip(*coord_meshes)]

    return coord_meshes

def get_angle(vec1: np.array, vec2: np.array, in_deg: bool=False, axis: int=None) -> np.array:
    """ Get the angle between two vectors vec1 and vec2.
    Both vec1 and vec2 must start at the origin (0,0).

    Parameters
    ----------
    vec1: an array representing the first vector of the angle

    vec2: an array representing the second vector of the angle

    in_deg: if True, will return the angle in degrees, else radians is used. Default is False.

    axis: if not None, will vectorize get_angle along the specified axis.
        Useful if vec1 and vec2 are MxNxD arrays where each element is a vector and the D
        axis is the dimensionality of the array (e.g. if vec1 and vec2 are 50x50x3 arrays
        then each vector is 3-dimensional and axis=2 should be used to vectorize get_angle,
        which will return a 50x50 array).
        Default is None (i.e. no vectorization).

    Returns
    -------
    An array of of angles.
    """

    ## a dot b = mag(a) * mag(b) * cos(theta)
    if axis == None:
        with np.errstate(invalid='raise'):
            try:
                rad = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            except FloatingPointError:
                rad = np.pi
    else:
        with np.errstate(invalid='ignore'):
            rad = np.arccos(np.sum(vec1 * vec2, axis=axis) / (np.linalg.norm(vec1, axis=axis) * np.linalg.norm(vec2, axis=axis)))
            rad[np.isnan(rad)] = np.pi
    ## note that this requires both of your vectors to start at the origin (i.e. (0,0)).
    ## also note that an invalid value runtimewarning is returned when two lines are
    ## perfectly parallel, so I have raised the warning and made the function return
    ## a value of pi (180deg) in the event that this is encountered. The angle between
    ## the vectors shouldn't be 0 since then the neighboring node and candidate end node
    ## for connection would be identical, which is a situation that is already handled
    ## elsewhere in the code.
    ## Also note that the axis can be set if you want to calculate angles for several
    ## vectors where vec1 and vec2 are both arrays of vectors with the vector oriented
    ## along 'axis'. E.g. an M x M array of vectors defined by points in (x,y) would
    ## be an M x M x 2 array == vec1. You could calculate the angle between all vectors
    ## then in vec1 and another vector vec2 (vec2 == a 1 x 1 x 2 array) by setting
    ## 'axis = 2' (since each vector is defined along axis 2).

    return np.rad2deg(rad) if in_deg else rad


def rasterize_edges_between_nodes(node_coord_pairs: list, arr_to_draw_on: np.array, label_lines: bool=False) -> np.array:
    """
    Takes a list of paired coordinates and an array and draws rasterized versions of
    the lines between the paired coordinates.

    Parameters
    ----------
    node_coord_pairs: list
        A list of paired coordinates to draw lines between, where one coordinate is
        the start of the line and the other is the end of the line.
        e.g.
        [((z1,y1,x1), (z2,y2,x2)),
         ((z3,y3,x3), (z4,y4,x4)),
         ...
         ]

    arr_to_draw_on: np.array
        The array where edges between node_coord_pairs are to be drawn;
        array shape must be consistent with the node coordinate pairs
        e.g. if coordinates are in (z, y, x) then the array should be
        an array with 3 dimensions.

    label_lines: bool
        Option where each line is given an integer value equal to its
        index + 1 in the node_coord_pairs list.
        Default is False.

    Returns
    -------
    The array that was drawn on if label_lines == False.
    If label_lines == True then a dictionary of labels and associated pixel locations
    is returned.
    """

    lines = {i+1: draw.line_nd(*node_coord_pairs[i], endpoint=True) for i in range(len(node_coord_pairs))}

    label_dict = {}
    ## We sort lines from largest to smallest so that the smaller ones are not completely overwritten
    ## in the event that a large and a small line have the same indices
    for label in sorted(lines, key=lambda x: len(list(zip(*lines[x]))), reverse=True):
        locs = lines[label]
        label_dict[label] = locs
        arr_to_draw_on[(*locs,)] = label if label_lines else True
    label_dict = {label: label_dict[label] for label in sorted(label_dict.keys())}

    return (arr_to_draw_on, label_dict) if label_lines else arr_to_draw_on

def build_vector(stop_position, start_position):
    vec = stop_position - start_position
    return vec

def calculate_neighbor_node_metrics(binary_image: np.array, intensity_image: np.array=None, VERBOSE=True) -> dict:
    """
    Takes a binary image representation of one or more structures that look
    approximately dendritic, filamentous, or network-like and creates a node
    and edge representation of the binary image to calculate angles between
    lines connecting neighboring nodes and a horizontal line as well as the
    lengths of those lines. Also calculates the edge lengths and intensities
    at the edges of an intensity_image is provided.
    Note that the edge lengths and local curvatures are not being used to
    calculate angles, only node-to-neighboring-node lines.

    Parameters
    ----------
    binary_image: np.array
        A binary array.
    
    intensity_image: np.array (optional)
        If provided, this image will be passed to to 'skimage.measure'.
        Default is None.

    Returns
    -------
    home_nodes_filtered: list
        The labels of the nodes of origin used to build a line.

    neighbor_nodes_filtered: list
        The labels of the end node used to build a line.

    dists_filtered: list
        The linear distance between home_nodes_filtered[index] and 
        neighbor_nodes_filtered[index].

    angles_filtered: list
        The angle between the line formed by home_nodes_filtered[index]
        to neighbor_nodes_filtered[index] and a horizontal line)

    node_label_pairs: list
        A list of node labels with home_nodes and neighbor_nodes paired together;
        note that if there a node has multiple neighbors it will show up multiple 
        times e.g. if n1 is connected to n2 and n3 then [(n1, n2), (n1, n3), ...].

    node_coord_pairs: list
        A list of the node centroids for the nodes in node_label_pairs in the same order.
    
    edge_metrics: list of dicts

    NOTE: home_nodes_filtered, neighbor_nodes_filtered, dists_filtered, and angles_filtered have the
    same indexing order (i.e. you can build a table directly from these lists), and node_label_pairs
    indices directly match those of node_coord_pairs, but the indices of home_nodes, neighbor_nodes,
    dists and angles do not match node_label_pairs or node_coord_pairs.

    TODO: this function should be split up into 2 functions: 1 for getting angles and
    distances, and another for getting edge lengths, fluorescences, etc...
    """
    ## convert cleaned up threshold of cadherin signal to nodes and edges
    nodes, edges, skel, conn = arr2graph(binary_image)
    del skel, conn # remove unused images to save on memory

    ## construct lines between all nodes
    node_props = measure.regionprops(nodes)
    node_labels, node_centroids = zip(*[(n.label, n.centroid) for n in node_props])

    print(f'    -- getting home node and neighboring node centroids') if VERBOSE else None
    node_label_grid1, node_label_grid2 = np.meshgrid(node_labels, node_labels, indexing='ij')
    ## construct vectors from the node centroids
    vec_nodes = build_vector(*numpy_mesh_coords(node_centroids, node_centroids, indexing='ij'))
    node_labels_index_dict = dict(zip(node_labels, range(len(node_labels))))
    print(f'    -- array shape = {vec_nodes.shape}') if VERBOSE else None

    print(f'    -- calculating distances of lines between neighboring nodes') if VERBOSE else None
    dists = np.linalg.norm(vec_nodes, axis=2)

    print(f'    -- calculating angles of lines between neighboring nodes') if VERBOSE else None
    ## determine angle of these lines relative to the horizontal
    ## (fluid flow direction is horizontal)
    ## construct a horizontal vector for reference
    ## indexing is ij, not xy, therefore (0,1) is horizontal
    vec_horizontal = np.array((0,1), ndmin=3)
    ## calculate angles between node-node lines and the horizontal line
    angles = get_angle(vec_horizontal, vec_nodes, in_deg=False, axis=2)

    ## since we are only measuring angles with the purpose of determining if a node-node
    ## connection is parallel or perpendicular, we need to fold all angles into the range
    ## 0-90. Currently the angles range from 0-180. This should reflect angles between
    ## 90-180 to be between 0-90
    angles[angles > np.pi/2] = abs(angles[angles > np.pi/2] - np.pi)

    print(f'    -- getting node neighbors') if VERBOSE else None
    ## get the node neighbors
    node_neighbors_edgelabs, edge_neighbors_nodelabs, node_neighbors_nodelabs = get_neighbor_nodes_and_edges(nodes, edges)
    del node_neighbors_edgelabs # remove unused images to save on memory

    ## create a connectivity matrix mask
    print(f'    -- creating node connectivity mask') if VERBOSE else None
    neighbors_mask = np.zeros(dists.shape, dtype=bool)
    ## node == i, neighbor == j
    node_neighbors_labels_list = [(node, neigh) for node, neighbors in node_neighbors_nodelabs for neigh in neighbors]
    node_neighbors_indices_list = [(node_labels_index_dict[node], node_labels_index_dict[neigh]) for node, neigh in node_neighbors_labels_list]
    neighbors_mask[tuple(zip(*node_neighbors_indices_list))] = True

    ## remove the top right diagonal half of the mask since they are just duplicates
    ## of the bottom half
    neighbors_mask_oneway = np.tril(neighbors_mask)

    ## filter the node-node distance and angle arrays so that only connected nodes are finite
    print(f'    -- filtering out unconnected node pairs') if VERBOSE else None
    home_nodes_filtered = node_label_grid1[neighbors_mask_oneway]
    neighbor_nodes_filtered = node_label_grid2[neighbors_mask_oneway]
    dists_filtered = dists[neighbors_mask_oneway]
    angles_filtered = angles[neighbors_mask_oneway]

    ## list the paired node labels and node coordinates for later use
    node_label_pairs = [(node, neigh) for node, neighbors in node_neighbors_nodelabs for neigh in neighbors]
    node_lab_coord_dict = dict(zip(node_labels, node_centroids))
    node_coord_pairs = [(node_lab_coord_dict[node], node_lab_coord_dict[neigh]) for node, neigh in node_label_pairs]

    ## calculate edge metrics
    node_neighbors_edgelabs, edge_neighbors_nodelabs, node_neighbors_nodelabs = get_neighbor_nodes_and_edges(nodes, edges, as_dict=True)
    extra_region_props = (get_length, intensity_std, intensity_median, intensity_pct25, intensity_pct75)
    edge_props = measure.regionprops(edges, intensity_image, extra_properties=extra_region_props)
    for prop in edge_props:
        try:
            prop.node_pair = edge_neighbors_nodelabs[prop.label]
        except IndexError:
            print(prop.label)

    node_pairs_filtered = list(zip(home_nodes_filtered, neighbor_nodes_filtered))
    edge_props_filtered = [[prop for prop in edge_props if set(prop.node_pair) == set(pair)] for pair in node_pairs_filtered]

    node_pair_labels = []
    node_pair_coords = []
    connecting_edge_labels = []
    edge_num_pixels = []
    edge_length = []
    edge_fluorescence_mean = []
    edge_fluorescence_std = []
    edge_fluorescence_median = []
    edge_fluorescence_min = []
    edge_fluorescence_pct25 = []
    edge_fluorescence_pct75 = []
    edge_fluorescence_max = []
    for edge_props in edge_props_filtered:
        node_pair_labels.append([prop.node_pair for prop in edge_props])
        node_pair_coords.append([tuple([node_lab_coord_dict[node] for node in prop.node_pair]) for prop in edge_props])
        connecting_edge_labels.append([prop.label for prop in edge_props])
        edge_num_pixels.append([prop.num_pixels for prop in edge_props])
        edge_length.append([prop.get_length for prop in edge_props])
        edge_fluorescence_mean.append([prop.intensity_mean for prop in edge_props])
        edge_fluorescence_std.append([prop.intensity_std for prop in edge_props])
        edge_fluorescence_median.append([prop.intensity_median for prop in edge_props])
        edge_fluorescence_min.append([prop.intensity_min for prop in edge_props])
        edge_fluorescence_pct25.append([prop.intensity_pct25 for prop in edge_props])
        edge_fluorescence_pct75.append([prop.intensity_pct75 for prop in edge_props])
        edge_fluorescence_max.append([prop.intensity_max for prop in edge_props])

    metrics = {'node_pair_labels': node_pair_labels,
               'node_pair_centroids': node_pair_coords,
               'distances': dists_filtered,
               'angles': angles_filtered,
               'edge_labels': connecting_edge_labels,
               'edge_num_pixels': edge_num_pixels,
               'length (px)': edge_length,
               'fluor_mean (au)': edge_fluorescence_mean,
               'fluor_std (au)': edge_fluorescence_std,
               'fluor_median (au)': edge_fluorescence_median,
               'fluor_min (au)': edge_fluorescence_min,
               'fluor_max (au)': edge_fluorescence_max,
               'fluor_pct25 (au)': edge_fluorescence_pct25,
               'fluor_pct75 (au)': edge_fluorescence_pct75,
               }

    return metrics

def intensity_std(region_mask: np.array, intensity_image: np.array) -> float:
    """This function is designed to be passed to the extra_properties argument
    of skimage.measure.regionprops.
    It will return the standard deviation of the intensity of the image within
    the label of the region."""
    region_intensity_std = np.std(intensity_image[region_mask])
    return region_intensity_std

def intensity_median(region_mask: np.array, intensity_image: np.array) -> float:
    """This function is designed to be passed to the extra_properties argument
    of skimage.measure.regionprops.
    It will return the median of the intensity of the image within the label
    of the region."""
    region_intensity_median = np.median(intensity_image[region_mask])
    return region_intensity_median

def intensity_pct25(region_mask: np.array, intensity_image: np.array) -> float:
    """This function is designed to be passed to the extra_properties argument
    of skimage.measure.regionprops.
    It will return the 25th percentile of the intensity of the image within the
    label of the region."""
    region_intensity_pct25 = np.percentile(intensity_image[region_mask], q=25)
    return region_intensity_pct25

def intensity_pct75(region_mask: np.array, intensity_image: np.array) -> float:
    """This function is designed to be passed to the extra_properties argument
    of skimage.measure.regionprops.
    It will return the 75th percentile of the intensity of the image within the
    label of the region."""
    region_intensity_pct75 = np.percentile(intensity_image[region_mask], q=75)
    return region_intensity_pct75

def walk_the_line(skel: np.array, max_num_pixels: int=None, bidirectional=True) -> tuple:
    """
    Takes a thinned or skeletonized binary line with 2 ends and no branches
    and returns the coordinates of the line ordered from endpoint to endpoint
    with the corresponding distance between the coordinates.
    Skeletonizations can be made with e.g. skimage.morphology.skeletonize and
    a node and edge finding algorithm (such as the function arr2graph here)
    can be used to split up the skeletonization into edge pieces with only
    two ends and no branches which are all connected together by nodes.
    Originally designed to work with the edges output from the 'arr2graph'.

    NOTE: This function has only been tested for 2D and 3D images, and likely
    will not work on arrays of higher dimensions.
    NOTE: Beware that an argument for checking if max_num_pixels exceeds the
    length of skel has not been implemented, and the current behavior is to
    count until the end of the line if it is shorter than max_num_pixels, not
    raise an error.

    Parameters
    ----------
    skel: np.array
        A thinned or skeletonized binary line with 2 ends and no branches.
    
    max_num_pixels: int
        How many pixels to move from a start point if specified. Useful for
        walking a defined number of pixels away from the ends of the lines
        (e.g. find the n-th pixel away from the endpoint), which can then be
        used to construct a vector from the endpoint to the n-th pixel away
        from the end point (this vector could approximate the curvature in
        part of the line).
        Default is to walk the whole line.

    bidirectional: bool
        If True, will return the ordered line coordinates starting from both
        ends of the line (i.e. the 'forward' and 'reverse' ordered coordinates),
        otherwise will return the ordered coordinates using only one endpoint as
        a starting point. Calculating the order of the coordinates using both
        endpoints as starting points takes twice as long as calculating only one.
        Default is True.
        
    Returns
    -------
    (line1, line2) or (line1,): tuple
        The ordered coordinates of the line from endpoint to endpoints structured
        as a dictionary as follows
            {line_first_coordinate: {next_closest_connected_coordinate: distance_to_closest_coordinate},
             ...,
             second_last_line_coordinate: {last_line_coordinate: distance_from_second_last_to_last_coordinate},
             last_line_coordinate: {}}
        Length of line1 and line2 should each be equal to the number of True pixels in skel.
        Returns (line1, line2) if bidirectional=True, else returns (line1,).
    """

    img_dim = skel.ndim
    max_num_pixels = max_num_pixels or np.count_nonzero(skel)

    coords = list(zip(*np.where(skel)))

    if len(coords) < 2:
        line1 = {(coords)[-1]: {}}
        if bidirectional:
            line2 = {(coords)[-1]: {}}
        else:
            pass
        pass

    else:
        coords1, coords2 = numpy_mesh_coords(coords, coords, indexing='xy')
        dists = np.linalg.norm(coords2 - coords1, axis=2)

        # conn1 = dists == 1
        # conn2 = dists == np.sqrt(2)
        # # conn3 = dists == np.sqrt(3)
        # conn_all = conn1 + conn2
        conns = [dists == np.sqrt(dim) for dim in range(1, img_dim+1)]
        conn_all = sum(conns).astype(bool)

        ## now mask the array
        dists = np.ma.masked_array(data=dists, mask=dists==0)

        edges_from_dist = [np.all(dists[(conn_all[i,:] * (conn_all[i,:] + conn_all)) * (conn_all[i,:] * (conn_all[i,:] + conn_all)).T] > np.sqrt(img_dim)) for i in range(len(conn_all))]

        edges_from_dist = np.array([x if x else False for x in edges_from_dist])


        edge_conn = np.array([np.count_nonzero(conn_arr, axis=1) == 2 for conn_arr in conns])
        edge_anticonn = np.array([np.count_nonzero(conn_arr, axis=1) == 0 for conn_arr in conns])

        edges_from_conn = sum([edge_conn[i,:] * ~(sum([~edge_anticonn[j] for j in range(len(edge_anticonn)) if j != i]).astype(bool)) for i in range(len(conns))]).astype(bool)
        edges_from_conn = edges_from_conn + (np.count_nonzero(conn_all, axis=1) > img_dim)

        ## this line of code will check the connectivity of the pixels
        ## connected to those found in maybe_nodes and if any of these
        ## connected pixels have both of THEIR neighbors in conn1, then
        ## we can safely say that the original pixel in maybe_nodes in
        ## question is in fact a node. This is because an edge pixel
        ## can't be only connected to 2 other pixels without one of 
        ## those connected neighbors being an edge pixel connected to
        ## two other pixels in the shape of an "L"

        edges = edges_from_dist + edges_from_conn
        nodes = ~edges

        assert(np.count_nonzero(nodes) == 2)

        ## now that we know which coordinates are nodes or edges,
        ## we can spatially order them (i.e. walk from one node
        ## along the edge of the line to the other node)

        ## starting from one node, move to the closest neighbour
        ## create a dictionary of which coordinates are next to which

        conns = {tuple(coords2[i,0].tolist()):
                    dict(zip([tuple(x) for x in coords1[i, conn_all[i]].tolist()], dists[i, conn_all[i]].tolist()))
                    for i in range(len(conn_all))}
        node_coords = [tuple(c) for c in np.asarray(coords)[nodes,:]]

        ## pick a starting position from one of the 2 endpoints of the coordinates
        ## (the choice is arbitrary)
        curr_node = node_coords[-1]
        visited_coords = [curr_node]
        line1 = {}

        for count in range(max_num_pixels):
            line1[curr_node] = {n: conns[curr_node][n] for n in conns[curr_node]
                                if conns[curr_node][n] == min([conns[curr_node][k] for k in conns[curr_node].keys() if k not in visited_coords], default=[])
                                and n not in visited_coords}
            if line1[curr_node]:
                curr_node = tuple(line1[curr_node].keys())[-1]
                visited_coords.append(curr_node)
            else:
                break

        if bidirectional:
            curr_node = node_coords[-2]
            visited_coords = [curr_node]
            line2 = {}

            for count in range(max_num_pixels):
                line2[curr_node] = {n: conns[curr_node][n] for n in conns[curr_node]
                                if conns[curr_node][n] == min([conns[curr_node][k] for k in conns[curr_node].keys() if k not in visited_coords], default=[])
                                and n not in visited_coords}
                if line2[curr_node]:
                    curr_node = tuple(line2[curr_node].keys())[-1]
                    visited_coords.append(curr_node)
                else:
                    break
        else:
            pass

    return (line1, line2) if bidirectional else (line1,)


def get_length(skel: np.array, max_num_pixels: int=None) -> float:
    """
    Returns the length of a rasterized binary line.
    The rasterized binary line must 2 ends and no branches.
    Originally designed to work with the edges output from
    arr2graph (see walk_the_line for more details).

    NOTE: This function has only been tested for 2D and 3D images, and likely
    will not work on arrays of higher dimensions.
    NOTE: Beware that an argument for checking if max_num_pixels exceeds the
    length of skel has not been implemented, and the current behavior is to
    count until the end of the line if it is shorter than max_num_pixels, not
    raise an error.

    Parameters
    ----------
    skel: np.array
        A thinned or skeletonized binary line with 2 ends and no branches.
    
    max_num_pixels: int
        How many pixels to move from a start point if specified. Useful for
        walking a defined number of pixels away from the ends of the lines
        (e.g. find the n-th pixel away from the endpoint), which can then be
        used to construct a vector from the endpoint to the n-th pixel away
        from the end point (this vector could approximate the curvature in
        part of the line).
        Default is to walk the whole line.

    Returns
    -------
    length: float
        The length of the line.
    """
    # get the coordinates of the line ordered from end to end and the
    # distances from one coordinate to the next
    line, = walk_the_line(skel, max_num_pixels, bidirectional=False)
    # get a list of the distances from one coordinate to the next
    dists = [line[startpoints][endpoints] for startpoints in line for endpoints in line[startpoints]]
    # return the total of the distances as the length
    length = sum(dists)

    return length
