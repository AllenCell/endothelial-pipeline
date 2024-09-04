import numpy as np
from skimage import filters
from skimage import measure
from skimage import draw
from skimage import morphology



def arr2graph(arr):
    """Will take a binary image array and return the nodes \
    and edges as well as their connections. """

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


def get_neighboring_labels(home_img, labeled_neighbors_img, bad_neighbors=None):
    """home_img will be made binary (can be an image where only a particular label was
    chosen by home_img == lab)
    bad_neighbors argument lets you choose labels in labeled_neighbors_img to exclude
    from result (e.g. 0 is often background, so may want to exclude 0)"""
    if home_img.ndim == 2:
        footprint = morphology.square(3)
    elif home_img.ndim == 3:
        footprint = morphology.cube(3)
    neighbors = [*np.unique(morphology.binary_dilation(home_img, footprint=footprint) * labeled_neighbors_img)]
    if bad_neighbors:
        # neighbors = [tuple([n for n in ns if n not in np.unique(bad_neighbors)]) for ns in neighbors]
        neighbors = [n for n in neighbors if n not in np.unique(bad_neighbors)]
    return tuple(neighbors)


def expand_bbox(bbox, ndim):
    big_bbox = (tuple((np.array(bbox[0:ndim]) - 0.5).astype(int)), tuple(np.array(bbox[ndim:2*ndim]) + 1))
    return big_bbox


def get_windows(img_lab): #labeled_img
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


def get_neighbor_nodes_and_edges(nodes_lab, edges_lab, bad_neighbors=[0], as_dict=False):

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


def numpy_mesh_coords(coord1_ls, coord2_ls, indexing='ij', return_indiv_coord_meshes=False):
    """Coordinate lists are lists of tuples, e.g.
    [(z1, y1, x1), (z2, y2, x2), ...]"""

    coords1 = zip(*coord1_ls)
    coords2 = zip(*coord2_ls)

    coords = list(zip(coords1, coords2))
    coord_meshes = [np.meshgrid(*coord_ax, indexing=indexing) for coord_ax in coords]

    if not return_indiv_coord_meshes:
        coord_meshes = [np.dstack(coords) for coords in zip(*coord_meshes)]

    return coord_meshes

def get_angle(vec1, vec2, in_deg=False, axis=None):
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


def rasterize_edges_between_nodes(node_coord_pairs, arr_to_draw_on, label_lines=False):
    """
    node_coord_pairs = [((z1,y1,x1), (z2,y2,x2)),
                        ((z3,y3,x3), (z4,y4,x4)),
                        ...
                        ]
    arr_to_draw_on = array where edges between node_coord_pairs are to be drawn;
                        array shape must be consistent with the node coordinate pairs
                        e.g. if coordinates are in (z, y, x) then the array should be
                        ## an array with 3 dimensions
    label_lines = option where each line is given an integer value equal to its
                    index + 1 in the node_coord_pairs list
    function will return both the array and a list of labels if label_lines == True,
    otherwise will return just the array if label_lines == False.
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
