import logging
import re
from collections.abc import Callable
from os import scandir
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from dask.array import Array
from skimage import draw, filters, graph, measure, morphology, segmentation

from endo_pipeline.library.process.general_image_preprocessing import ImageProcessingArgs

logger = logging.getLogger(__name__)


def arr2graph(
    arr: np.ndarray, closing_step: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Will take a binary image array showing a network-like structure
    and return the labeled versions of the nodes, edges, skeletons and
    pixel connectivity in that order. The connectivity equal to the
    dimensionality of arr (if 2D then a connectivity of 2 or a 3x3 square
    is used, if 3D then a connectivity of 3 or a 3x3x3 cube is used).

    Parameters
    ----------
    arr: np.ndarray
        A binary 2D or 3D numpy array representing an image with dendritic, branching,
        tree-like, or network-like structures.

    closing_step: bool
        Whether to do binary closing with connectivity equal to the array dimensionality
        on 'arr' before skeletonization and calculation of edges and nodes.
        The closing step can be useful for connecting small break in the network, but it
        will also turn very small curves into branching points (i.e. it will introduce a
        node and edge there). Therefore, if 'arr' is expected / known to be a completely
        closed network then closing_step should be set to False.
        An example of a completely closed network would be if 'arr' were borders from a
        segmentation.

    Returns
    -------
    nodes_lab: np.ndarray
        The nodes in arr where each node has a unique label as an array of the same
        shape as arr.

    edges_lab: np.ndarray
        The edges in arr where each edge has a unique label as an array of the same
        shape as arr.

    skels_lab: np.ndarray
        The skeletonization of arr where unconnected skeletons have unique labels as
        an array of the same shape as arr.

    conn: np.ndarray
        The connectivity of each pixel in arr as an array of the same shape as arr.
    """

    ## Make sure that the array is either 2D or 3D
    assert arr.ndim == 2 or arr.ndim == 3, "Input array must be 2D or 3D."

    if arr.ndim == 2:
        footprint = morphology.square(3)
    elif arr.ndim == 3:
        footprint = morphology.cube(3)

    ## Fill any tiny holes
    arr = morphology.binary_closing(arr, footprint=footprint) if closing_step else arr
    skel = morphology.skeletonize(arr).astype(bool)
    ## skeletonize above will make your array int8 dtype, and
    ## will make True == 255, but I want it to be 1, so I will
    ## force it to be bool, hence the .astype above.

    ## Converting the bool to int now does not make
    ## True -> 255, instead True -> 1 (which is what I want):
    ## Transform the skeletonized array into one where each
    ## pixel has a value equal to the number of non-zero
    ## immediate neighbors plus itself
    ## the * skel is to re-skeletonize the rank sum
    conn = filters.rank.pop(skel.astype(np.uint8), footprint=footprint, mask=skel) * skel
    # This produces an array with the following values
    # (which is why I insisted on having the skeletonized array
    # have only 0s and 1s as values):
    # conn == 1,2 -> node (isolated point)
    # conn == 2 -> node (end point)
    # conn == 3 -> edge
    # conn >= 4 -> node (branch point)

    ## Label those endpoints, edges, and branchpoints (this is
    ## to get the connections between edges and nodes later on):
    edges_arr = conn == 3
    nodes_arr = (conn == 1) + (conn == 2) + (conn >= 4)

    ## There can be both isolated nodes (a single pixel in space)
    ## and isolated edges (a closed loop in space)
    ## how do you uniquely define such a graph?
    ## Both edges and nodes need their own labels.
    nodes_lab = morphology.label(nodes_arr, connectivity=arr.ndim)
    edges_lab = morphology.label(edges_arr, connectivity=arr.ndim)
    skels_lab = morphology.label(skel, connectivity=arr.ndim)

    return nodes_lab, edges_lab, skels_lab, conn


def get_neighboring_labels(
    home_img: np.ndarray,
    labeled_neighbors_img: np.ndarray,
    bad_neighbors: list[Any] | None = None,
) -> tuple:
    """
    home_img will be made binary (can be an image where only a particular label was
    chosen by home_img == lab)
    bad_neighbors argument lets you choose labels in labeled_neighbors_img to exclude
    from result (e.g. 0 is often background, so may want to exclude 0).

    Parameters
    ----------
    home_img: np.ndarray
        A binary array of the region you want to get the neighboring regions of.

    labeled_neighbors_img: np.ndarray
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
    neighbors = [
        *np.unique(
            morphology.binary_dilation(home_img, footprint=footprint) * labeled_neighbors_img
        )
    ]
    if bad_neighbors:
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

    big_bbox = (
        tuple((np.array(bbox[0:ndim]) - 0.5).astype(int)),
        tuple(np.array(bbox[ndim : 2 * ndim]) + 1),
    )

    return big_bbox


def get_windows(img_lab: np.ndarray) -> zip:  # labeled_img
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
    lab_labs, lab_bbox = zip(*[(lab.label, lab.bbox) for lab in img_lab_props], strict=False)

    # Apparently Python now allows your upper slice range to exceed bounds, and instead
    # will just return the values within range.
    # Grab a bbox that is 1 pixel wider on each edge of each axis:
    lab_bbox_big = [expand_bbox(bbox, ndim) for bbox in lab_bbox]

    # Create slicing windows of these expanded bboxes:
    windows = [[slice(*i) for i in list(zip(*bb, strict=False))] for bb in lab_bbox_big]

    # zip the labels and windows together
    lab_windows = zip(lab_labs, windows, strict=False)

    return lab_windows


def get_neighbor_nodes_and_edges(
    nodes_lab: np.ndarray,
    edges_lab: np.ndarray,
    bad_neighbors: list = [0],
    as_dict: bool = False,
) -> tuple:
    """
    Takes a labeled array of nodes and a labeled array of edges and returns
    a list or dict of which nodes neighbor each node, which edges neighbor each node,
    and which nodes neighbor each edge.
    The reason both lists of nodes and lists of edges are returned is because it is
    possible for a node or edge to have no neighbors.
    This function is designed to work with the output of the arr2graph function.

    Parameters
    ----------
    nodes_lab: np.ndarray
        The labeled nodes.

    edges_lab: np.ndarray
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
    node_neighbors_edgelabs = [
        (
            label,
            get_neighboring_labels(
                nodes_lab[(*window,)] == label, edges_lab[(*window,)], bad_neighbors=bad_neighbors
            ),
        )
        for label, window in nodes_lab_windows
    ]
    edge_neighbors_nodelabs = [
        (
            label,
            get_neighboring_labels(
                edges_lab[(*window,)] == label, nodes_lab[(*window,)], bad_neighbors=bad_neighbors
            ),
        )
        for label, window in edges_lab_windows
    ]

    ## Use the combination of node_neighbors_edgelabs and edge_neighbors_nodelabs to
    ## find which nodes neighbor each other:
    # nodes_lab_props = measure.regionprops(nodes_lab, intensity_image=skel_lab)

    node_neighbors_nodelabs = []
    for x in node_neighbors_edgelabs:
        # Get which edges are connected to a particular node:
        node, edges = x
        # Iterate through the edge_neighbors and look for connected nodes
        # in the edge_neighbors_nodelabs list:
        node_neighbors_nodelabs.append(
            (node, [n for e, n in edge_neighbors_nodelabs if e in edges])
        )
    # Clean up the node list with node neighbors so that  there are no repeating node labels
    node_neighbors_nodelabs_unique = [
        (node, tuple(np.unique([n for ns in n_neighbors for n in ns])))
        for node, n_neighbors in node_neighbors_nodelabs
    ]
    # and also remove the "home node" from the node neighbors list to get the final cleaned up list:
    node_neighbors_nodelabs_clean = [
        (node, tuple([n for n in n_neighbors if n != node]))
        for node, n_neighbors in node_neighbors_nodelabs_unique
    ]

    if not as_dict:
        return (
            node_neighbors_edgelabs,
            edge_neighbors_nodelabs,
            node_neighbors_nodelabs_clean,
        )
    else:
        node_neighbors_edgelabs_dict = dict(node_neighbors_edgelabs)
        edge_neighbors_nodelabs_dict = dict(edge_neighbors_nodelabs)
        node_neighbors_nodelabs_dict = dict(node_neighbors_nodelabs_clean)
        return (
            node_neighbors_edgelabs_dict,
            edge_neighbors_nodelabs_dict,
            node_neighbors_nodelabs_dict,
        )


def numpy_mesh_coords(
    coord1_ls: list[Any] | tuple[Any],
    coord2_ls: list[Any] | tuple[Any],
    indexing: Literal["xy", "ij"] = "ij",
    return_indiv_coord_meshes: bool = False,
) -> list:
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

    assert (
        np.array(coord1_ls).ndim == np.array(coord2_ls).ndim <= 2
    ), "Coordinate lists must be 2D or 1D and have same dimensions."

    coord1_array = (
        np.array(coord1_ls) if np.array(coord1_ls).ndim == 2 else np.array(coord1_ls, ndmin=2).T
    )
    coord2_array = (
        np.array(coord2_ls) if np.array(coord2_ls).ndim == 2 else np.array(coord2_ls, ndmin=2).T
    )

    coords1 = zip(*coord1_array, strict=False)
    coords2 = zip(*coord2_array, strict=False)

    coords = list(zip(coords1, coords2, strict=False))
    coord_meshes = [np.meshgrid(*coord_ax, indexing=indexing) for coord_ax in coords]

    if not return_indiv_coord_meshes:
        return [np.dstack(coords) for coords in zip(*coord_meshes, strict=False)]

    return coord_meshes


def get_angle(
    vec1: np.ndarray, vec2: np.ndarray, in_deg: bool = False, axis: int | None = None
) -> np.ndarray:
    """Get the angle between two vectors vec1 and vec2.
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
    if axis is None:
        with np.errstate(invalid="raise"):
            try:
                rad = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            except FloatingPointError:
                rad = np.pi
    else:
        with np.errstate(invalid="ignore"):
            rad = np.arccos(
                np.sum(vec1 * vec2, axis=axis)
                / (np.linalg.norm(vec1, axis=axis) * np.linalg.norm(vec2, axis=axis))
            )
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


def rasterize_edges_between_nodes(
    node_coord_pairs: list, arr_to_draw_on: np.ndarray, label_lines: bool = False
) -> tuple[np.ndarray, dict | None]:
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

    arr_to_draw_on: np.ndarray
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

    lines = {
        i + 1: draw.line_nd(*node_coord_pairs[i], endpoint=True)
        for i in range(len(node_coord_pairs))
    }

    label_dict = {}
    ## We sort lines from largest to smallest so that the smaller ones are not completely
    ## overwritten in the event that a large and a small line have the same indices
    for label in sorted(lines, key=lambda x: len(list(zip(*lines[x], strict=False))), reverse=True):
        locs = lines[label]
        label_dict[label] = locs
        arr_to_draw_on[(*locs,)] = label if label_lines else True
    label_dict = {label: label_dict[label] for label in sorted(label_dict.keys())}

    return (arr_to_draw_on, label_dict) if label_lines else (arr_to_draw_on, None)


def build_vector(stop_position: np.ndarray, start_position: np.ndarray) -> np.ndarray:
    vec = stop_position - start_position
    return vec


def calculate_region_border_metrics(
    binary_image: np.ndarray,
    intensity_image: np.ndarray | Array | None = None,
    labeled_image: np.ndarray | Array | None = None,
) -> list:
    """
    Takes a binary image representation of one or more structures that look
    approximately dendritic, filamentous, or network-like and creates a node
    and edge representation of the binary image to calculate angles between
    lines connecting neighboring nodes and a horizontal line as well as the
    lengths of those lines.

    Also calculates the edge lengths and intensities at the edges of an
    intensity_image if provided.
    Note that the edge lengths and local curvatures are not being used to
    calculate angles, only node-to-neighboring-node lines.

    If labeled_image is provided then metrics for each region in labeled_image
    will be returned as a second dictionary of lists, including associated
    node labels, edge labels, and paired node labels.

    **Neighbor node metrics**

    ======================  ===================================================================
    Metric                  Description
    ======================  ===================================================================
    node_pair_labels        labels of (origin, neighbor) nodes used to build a line
    node_pair_centroids     centroids of (origin, neighbor) nodes used to build a line
    distances               linear distance between node_pair_centroids
    angles                  angle between the line formed by node_pair_centroids and horizontal
    edge_labels             labels of the edges in binary_image connecting the paired nodes
    edge_num_pixels         number of pixels that constitute each edge
    length (px)             length of each edge in pixels
    fluor_mean (au)         mean fluorescence of intensity_image at an edge
    ======================  ===================================================================

    **Labeled image metrics**

    ============================  =================================================================
    Metric                        Description
    ============================  =================================================================
    cell_label                    labels of the regions in labeled_image
    cell_centroid                 centroids of the regions in labeled_image
    cell_area (px**2)             areas of the regions in labeled_image
    cell_perimeter (px)           perimeters of the regions in labeled_image
    cell_solidity                 solidities of the regions in labeled_image
    cell_eccentricity             eccentricities of the regions in labeled_image
    cell_orientation              orientations of the regions in labeled_image
    cell_fluorescence_mean (au)   mean fluorescence of intensity_image for each region
    edge_labels                   labels of the edges that touch each region in labeled_image
    node_labels                   labels of the nodes that touch each region in labeled_image
    edge_fluorescences (au)       list of fluorescence values at the edges of each region
    node_fluorescences (au)       list of fluorescence values at the nodes of each region
    node_pair_labels              labels of the nodes at the end of each edge label for each region
    ============================  =================================================================


    Parameters
    ----------
    binary_image: np.ndarray
        The binary array to be converted into an array of labeled nodes and labeled edges.

    intensity_image: np.ndarray (optional)
        If provided, this image will be passed to 'skimage.measure.regionprops' and used when
        measuring fluorescence intensities. If None, returned fluorescence lists will contain
        np.nans.
        Default is None.

    labeled_image: np.ndarray (optional)
        If provided, measurements using 'skimage.measure' of labeled_image will be made and
        associated with the node labels and edge labels that result from creating a node and
        edge representation of binary_image. If None then no measurements will be made and
        'None' will be returned instead of a dict of lists.
        Default is None.

    Returns
    -------
    :
        List of dictionaries containing neighbor node metrics and labeled image metrics.

    NOTE: The lists in each 'metrics' dict have the same indexing order (i.e. you can build a
        table directly from this dict via a pandas DataFrame).
    """

    ## if intensity_image is not provided then make a dummy channel full of np.nans
    intensity_image = (
        intensity_image
        if isinstance(intensity_image, np.ndarray)
        else np.full(binary_image.shape, np.nan)
    )

    ## convert cleaned up threshold of cadherin signal to nodes and edges
    nodes, edges, skel, conn = arr2graph(binary_image, closing_step=False)
    del skel, conn  # remove unused images to save on memory

    ## calculate neighbor node angles and distances
    neighbor_node_metrics = calculate_neighbor_node_metrics(
        binary_image, nodes, edges, intensity_image
    )  # -> list of dictionaries(?)

    ## associate edges with the labeled_image
    if isinstance(labeled_image, np.ndarray):
        labeled_image_metrics = calculate_labeled_image_metrics(
            binary_image, labeled_image, nodes, edges, intensity_image
        )
    else:
        labeled_image_metrics = None

    return [neighbor_node_metrics, labeled_image_metrics]


def calculate_labeled_image_metrics(
    binary_image: np.ndarray,
    labeled_image: np.ndarray,
    nodes: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    intensity_image: np.ndarray | None = None,
) -> dict:
    """
    Takes a binary image representation of one or more structures that look
    approximately dendritic, filamentous, or network-like and its node and
    edge representation (a representation will be created from binary_image
    if they are not provided) to generate region properties from
    skimage.measure.regionprops and associate them with the node labels,
    edge labels, and node pairs that surround each region in labeled_image.
    Ideally labeled_image is a confluent or space-filling segmentation and
    binary_image is simply the boundaries of this segmentation (produced
    from skimage.segmenation.find_boundaries(labeled_image)).
    Should also work with non-space-filling segmentations and a binary_image
    that is thicker than one produced by find_boundaries but this has not
    been tested and may result in unexpected outputs.

    Parameters
    ----------
    binary_image: np.ndarray
        The binary array to be converted into an array of labeled nodes and labeled edges.

    labeled_image: np.ndarray
        If provided, measurements using 'skimage.measure' of labeled_image will be made and
        associated with the node labels and edge labels arguments. If no node labels and edge
        labels arguments are provided a node and edge representation of binary_image will be
        created.

    nodes: np.ndarray (optional)
        An array of labeled nodes produced from binary_image using arr2graph.
        Will be generated from binary_image if not provided.
        Default is None.

    edges: np.ndarray (optional)
        An array of labeled edges produced from binary_image using arr2graph.
        Will be generated from binary_image if not provided.
        Default is None.

    intensity_image: np.ndarray (optional)
        If provided, this image will be passed to to 'skimage.measure.regionprops'.
        Default is None.

    Returns
    -------
    metrics: dict of lists
        cell_label: The labels of the regions in labeled_image.
        cell_centroid: The centroids of the regions in labeled_image.
        cell_area (px**2): The areas of the regions in labeled_image.
        cell_perimeter (px): The perimeters of the regions in labeled_image.
        cell_solidity: The solidities of the regions in labeled_image
        cell_eccentricity: The eccentricities of the regions in labeled_image.
        cell_orientation: The orientations of the regions in labeled_image.
        cell_fluorescence_mean (au): The mean fluorescence of intensity_image for each
            region in labeled_image (if intensity_image is provided).
            Other fluorescence measures include _std, _median, _min, _max, _pct25, and _pct75.
        edge_labels: The labels of the edges that touch each region in labeled_image.
        node_labels: The labels of the nodes that touch each region in labeled_image.
        node_pair_labels: The labels of the node pairs that are at the end of each edge label
            that touches each region in labeled_image.

    NOTE: The lists in 'metrics' have the same indexing order (i.e. you can build a table directly
        from this dict).
    """

    # ensure that binary_image is a boolean array
    assert binary_image.dtype == np.dtype(bool), "dtype of binary_image array must be bool."

    # remove any regions in seeds that overlap with the binary image
    # that is used to generate the nodes and edges representation
    seeds = labeled_image.copy()
    seeds *= ~binary_image

    # create the nodes and edges arrays if they were not provided
    logger.debug("- getting node and edge labels")
    nodes, edges, skel, conn = (
        (nodes, edges, None, None)
        if (isinstance(nodes, np.ndarray) and isinstance(edges, np.ndarray))
        else arr2graph(binary_image, closing_step=False)
    )
    del skel, conn  # remove unused images to save on memory

    # if intensity_image is not provided then make a dummy channel full of np.nans
    # so that measure.regionprops doesn't return an error when trying to measure
    # the fluorescence
    intensity_image = (
        intensity_image
        if isinstance(intensity_image, np.ndarray)
        else np.full(binary_image.shape, np.nan)
    )

    # get the node labels that define each edge
    logger.debug("- getting neighboring node information")
    node_neighbors_edgelabs, edge_neighbors_nodelabs, node_neighbors_nodelabs = (
        get_neighbor_nodes_and_edges(nodes, edges, as_dict=True)
    )

    # run a watershed using the labeled (minus any regions that overlap with binary_image)
    # image as seeds to find which parts of labels touch which edges
    logger.debug("- expanding labels in labeled_image to be adjacent to edges")
    regions = segmentation.watershed(
        np.logical_or(nodes, edges),
        markers=seeds,
        connectivity=1,  # labeled_image.ndim,
        mask=~np.logical_or(nodes, edges),
    ).astype(np.int32)
    # make the labeling of regions start after the biggest edge label
    logger.debug("- relabeling labeled_image")
    regions_offset = regions.copy()
    regions_offset[regions.astype(bool)] += edges.max()
    # combine the edges labels and the offset regions labels
    regions_offset += edges

    # create a RAG from the regions and find out which edge labels and connected to which
    # region labels
    logger.debug("- finding which edge labels touch which labeled_image regions")

    rag = graph.rag_boundary(
        regions_offset, np.zeros(labeled_image.shape, dtype=float), connectivity=1
    )
    # remove any connections to background (the background in this case would be any nodes
    # or unreachable areas)
    rag.remove_node(0) if 0 in rag.nodes else None

    logger.debug("- finding which region labels are neighbors")
    rag_of_labeled_image = graph.rag_boundary(
        labeled_image, np.zeros(labeled_image.shape, dtype=float), connectivity=1
    )
    # rag_of_labeled_image = graph.RAG(labeled_image)
    # remove any connections to background (the background in this case would be any nodes
    # or unreachable areas)
    rag_of_labeled_image.remove_node(0) if 0 in rag_of_labeled_image.nodes else None

    # map the labels in regions_offset to their original labels
    region_map = dict(
        zip(
            regions[regions.astype(bool)],
            regions_offset[regions.astype(bool)],
            strict=False,
        )
    )

    # get the region properties of the labels in regions
    logger.debug("- getting labeled_image region properties")
    extra_region_props = (
        intensity_std,
        intensity_median,
        intensity_pct25,
        intensity_pct75,
    )
    region_props = measure.regionprops(
        regions, intensity_image=intensity_image, extra_properties=extra_region_props
    )

    # add the neighbors of each region in regions
    logger.debug(
        "- adding node label and edge label information to region labeled_image properties"
    )
    logger.debug("- adding neighboring region information to labeled_image properties")

    for region in region_props:
        # include the neighbor labels if the label is an edge label (but not if it happens to
        # a label originating from labeled_image)
        neighbors = tuple(
            [
                neigh
                for neigh in rag.neighbors(region_map[region.label])
                if neigh not in region_map.values()
            ]
        )
        region.neighbors = neighbors
        # add the neighboring region labels of each region in the labeled_image
        region_neighbors = tuple(
            [
                neigh
                for neigh in rag_of_labeled_image.neighbors(region.label)
                if neigh != region.label
            ]
        )
        region.region_neighbors = region_neighbors

    # get the labels of the regions that touch the image borders
    border_labels = np.unique(
        ~segmentation.clear_border(labeled_image).astype(bool) * labeled_image
    )

    edge_props = measure.regionprops(edges, intensity_image=intensity_image)
    edge_props_dict = {prop.label: prop for prop in edge_props}
    node_props = measure.regionprops(nodes, intensity_image=intensity_image)
    node_props_dict = {prop.label: prop for prop in node_props}

    # create the output lists
    logger.debug("- generating dictionary of lists output")
    region_label = []
    region_centroid = []
    region_area = []
    region_perimeter = []
    region_solidity = []
    region_major_axis_length = []
    region_minor_axis_length = []
    region_eccentricity = []
    region_orientation = []
    region_fluor_mean = []
    region_fluor_std = []
    region_fluor_median = []
    region_fluor_min = []
    region_fluor_pct25 = []
    region_fluor_pct75 = []
    region_fluor_max = []
    neighboring_regions = []
    edge_labels = []
    node_labels = []
    node_pairs = []
    edge_fluor_list = []
    node_fluor_list = []
    is_border_region = []

    for prop in region_props:
        region_label.append(prop.label)
        region_centroid.append(prop.centroid)
        region_area.append(prop.area)
        region_perimeter.append(prop.perimeter)
        region_solidity.append(prop.solidity)
        region_major_axis_length.append(prop.major_axis_length)
        region_minor_axis_length.append(prop.minor_axis_length)
        region_eccentricity.append(prop.eccentricity)
        # add pi/2 to to orientation to make the orientation have 0 and pi
        # represent the X axis and pi/2 represent the Y axis (instead of
        # the default where 0 and pi represent the Y axis and pi/2 the X axis)
        region_orientation.append(prop.orientation + np.pi / 2)
        region_fluor_mean.append(prop.intensity_mean)
        region_fluor_std.append(prop.intensity_std)
        region_fluor_median.append(prop.intensity_median)
        region_fluor_min.append(prop.intensity_min)
        region_fluor_pct25.append(prop.intensity_pct25)
        region_fluor_pct75.append(prop.intensity_pct75)
        region_fluor_max.append(prop.intensity_max)
        neighboring_regions.append(prop.region_neighbors)
        edge_labels.append(prop.neighbors)
        node_neighbors = {node for edge in prop.neighbors for node in edge_neighbors_nodelabs[edge]}
        node_labels.append(node_neighbors)
        node_pairs.append([edge_neighbors_nodelabs[edge] for edge in prop.neighbors])
        is_border_region.append(prop.label in border_labels)
        edge_fluors = [
            intens
            for edge in prop.neighbors
            for intens in intensity_image[
                tuple(zip(*edge_props_dict[edge].coords, strict=True))
            ].tolist()
        ]
        edge_fluor_list.append(edge_fluors)
        node_fluors = [
            intens
            for node in node_neighbors
            for intens in intensity_image[
                tuple(zip(*node_props_dict[node].coords, strict=True))
            ].tolist()
        ]
        node_fluor_list.append(node_fluors)

    # create the output dictionary of lists
    metrics = {
        "cell_label": region_label,
        "cell_centroid": region_centroid,
        "cell_area (px**2)": region_area,
        "cell_perimeter (px)": region_perimeter,
        "cell_solidity": region_solidity,
        "major_axis_length": region_major_axis_length,
        "minor_axis_length": region_minor_axis_length,
        "cell_eccentricity": region_eccentricity,
        "cell_orientation": region_orientation,
        "cell_fluorescence_mean (au)": region_fluor_mean,
        "cell_fluorescence_std (au)": region_fluor_std,
        "cell_fluorescence_median (au)": region_fluor_median,
        "cell_fluorescence_min (au)": region_fluor_min,
        "cell_fluorescence_pct25 (au)": region_fluor_pct25,
        "cell_fluorescence_pct75 (au)": region_fluor_pct75,
        "cell_fluorescence_max (au)": region_fluor_max,
        "neighboring_cell_labels": neighboring_regions,
        "edge_labels": edge_labels,
        "node_labels": node_labels,
        "node_pair_labels": node_pairs,
        "edge_fluorescences (au)": edge_fluor_list,
        "node_fluorescences (au)": node_fluor_list,
        "touches_image_border": is_border_region,
    }

    return metrics


def calculate_neighbor_node_metrics(
    binary_image: np.ndarray,
    nodes: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    intensity_image: np.ndarray | None = None,
) -> dict:
    """
    Takes a binary image representation of one or more structures that look
    approximately dendritic, filamentous, or network-like and creates a node
    and edge representation of the binary image to calculate angles between
    lines connecting neighboring nodes and a horizontal line as well as the
    lengths of those lines. Also calculates the edge lengths and intensities
    at the edges of an intensity_image if provided.
    Note that the edge lengths and local curvatures are not being used to
    calculate angles, only node-to-neighboring-node lines.

    Parameters
    ----------
    binary_image: np.ndarray
        The binary array to be converted into an array of labeled nodes and labeled edges
        if 'nodes' argument and 'edges' argument are not provided.

    nodes: np.ndarray (optional)
        An array of labeled nodes produced from binary_image using arr2graph.
        Default is None.

    edges: np.ndarray (optional)
        An array of labeled edges produced from binary_image using arr2graph.
        Default is None.

    intensity_image: np.ndarray (optional)
        If provided, this image will be passed to to 'skimage.measure.regionprops'.
        Default is None.

    Returns
    -------
    metrics: dict of lists
        node_pair_labels: The labels of the nodes used to build a line with
            the order (origin_node, neighboring_node).
        node_pair_centroids: The centroids of the nodes used to build a line
            with the order (origin_node, neighboring_node)
        distances: The linear distance between node_pair_centroids.
        angles: The angle between the line formed by node_pair_centroids and a horizontal line.
        edge_labels: The labels of the edges in binary_image that connect the paired nodes.
        edge_num_pixels: The number of pixels that constitute each edge. Does not account for
            differences in distance based on connectivity (but 'length (px)' does).
        length (px): The length of each edge in pixels (N.B. this does not include the distance
            from the node to the edge).
        fluor_mean (au): The mean fluorescence of intensity_image at an edge if provided.
            Other measures for fluor include _std, _median, _min, _max, _pct25, and _pct75.

    NOTE: The lists in 'metrics' have the same indexing order (i.e. you can build a table directly
        from this dict).
    """

    # create the nodes and edges arrays if they were not provided
    nodes, edges, skel, conn = (
        (nodes, edges, None, None)
        if (isinstance(nodes, np.ndarray) and isinstance(edges, np.ndarray))
        else arr2graph(binary_image, closing_step=False)
    )
    del skel, conn  # remove unused images to save on memory

    # if intensity_image is not provided then make a dummy channel full of np.nans
    # so that measure.regionprops doesn't return an error when trying to measure
    # the fluorescence
    intensity_image = (
        intensity_image
        if isinstance(intensity_image, np.ndarray)
        else np.full(binary_image.shape, np.nan)
    )

    ## construct lines between all nodes
    node_props = measure.regionprops(nodes)
    node_labels, node_centroids = zip(*[(n.label, n.centroid) for n in node_props], strict=False)

    logger.debug("- getting home node and neighboring node centroids")
    node_label_grid1, node_label_grid2 = np.meshgrid(node_labels, node_labels, indexing="ij")
    ## construct vectors from the node centroids
    vec_nodes = build_vector(*numpy_mesh_coords(node_centroids, node_centroids, indexing="ij"))
    node_labels_index_dict = dict(zip(node_labels, range(len(node_labels)), strict=False))
    logger.debug(f"- array shape = {vec_nodes.shape}")

    logger.debug("- calculating distances of lines between neighboring nodes")
    dists = np.linalg.norm(vec_nodes, axis=2)

    logger.debug("- calculating angles of lines between neighboring nodes")
    ## determine angle of these lines relative to the horizontal
    ## (fluid flow direction is horizontal)
    ## construct a horizontal vector for reference
    ## indexing is ij, not xy, therefore (0,1) is horizontal
    vec_horizontal = np.array((0, 1), ndmin=3)
    ## calculate angles between node-node lines and the horizontal line
    angles = get_angle(vec_horizontal, vec_nodes, in_deg=False, axis=2)

    ## since we are only measuring angles with the purpose of determining if a node-node
    ## connection is parallel or perpendicular, we need to fold all angles into the range
    ## 0-90. Currently the angles range from 0-180. This should reflect angles between
    ## 90-180 to be between 0-90
    angles[angles > np.pi / 2] = abs(angles[angles > np.pi / 2] - np.pi)

    logger.debug("- getting node neighbors")
    ## get the node neighbors
    node_neighbors_edgelabs, edge_neighbors_nodelabs, node_neighbors_nodelabs = (
        get_neighbor_nodes_and_edges(nodes, edges)
    )
    del node_neighbors_edgelabs  # remove unused images to save on memory

    ## create a connectivity matrix mask
    logger.debug("- creating node connectivity mask")
    neighbors_mask = np.zeros(dists.shape, dtype=bool)
    ## node == i, neighbor == j
    node_neighbors_labels_list = [
        (node, neigh) for node, neighbors in node_neighbors_nodelabs for neigh in neighbors
    ]
    node_neighbors_indices_list = [
        (node_labels_index_dict[node], node_labels_index_dict[neigh])
        for node, neigh in node_neighbors_labels_list
    ]
    neighbors_mask[tuple(zip(*node_neighbors_indices_list, strict=False))] = True

    ## remove the top right diagonal half of the mask since they are just duplicates
    ## of the bottom half
    neighbors_mask_oneway = np.tril(neighbors_mask)

    ## filter the node-node distance and angle arrays so that only connected nodes are finite
    logger.debug("- filtering out unconnected node pairs")
    home_nodes_filtered = node_label_grid1[neighbors_mask_oneway]
    neighbor_nodes_filtered = node_label_grid2[neighbors_mask_oneway]
    dists_filtered = dists[neighbors_mask_oneway]
    angles_filtered = angles[neighbors_mask_oneway]

    ## list the paired node labels and node coordinates for later use
    node_lab_coord_dict = dict(zip(node_labels, node_centroids, strict=False))

    ## calculate edge metrics
    node_neighbors_edgelabs, edge_neighbors_nodelabs, node_neighbors_nodelabs = (
        get_neighbor_nodes_and_edges(nodes, edges, as_dict=True)
    )
    extra_region_props = (
        get_length,
        intensity_std,
        intensity_median,
        intensity_pct25,
        intensity_pct75,
    )
    edge_props = measure.regionprops(edges, intensity_image, extra_properties=extra_region_props)
    for prop in edge_props:
        try:
            prop.node_pair = edge_neighbors_nodelabs[prop.label]
        except IndexError:
            logger.info(prop.label)

    node_pairs_filtered = list(zip(home_nodes_filtered, neighbor_nodes_filtered, strict=False))
    edge_props_filtered = [
        [prop for prop in edge_props if set(prop.node_pair) == set(pair)]
        for pair in node_pairs_filtered
    ]

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
        node_pair_coords.append(
            [tuple([node_lab_coord_dict[node] for node in prop.node_pair]) for prop in edge_props]
        )
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

    metrics = {
        "node_pair_labels": node_pair_labels,
        "node_pair_centroids": node_pair_coords,
        "distances": dists_filtered,
        "angles": angles_filtered,
        "edge_labels": connecting_edge_labels,
        "edge_num_pixels": edge_num_pixels,
        "length (px)": edge_length,
        "fluor_mean (au)": edge_fluorescence_mean,
        "fluor_std (au)": edge_fluorescence_std,
        "fluor_median (au)": edge_fluorescence_median,
        "fluor_min (au)": edge_fluorescence_min,
        "fluor_pct25 (au)": edge_fluorescence_pct25,
        "fluor_pct75 (au)": edge_fluorescence_pct75,
        "fluor_max (au)": edge_fluorescence_max,
    }

    return metrics


def intensity_std(region_mask: np.ndarray, intensity_image: np.ndarray) -> float:
    """This function is designed to be passed to the extra_properties argument
    of skimage.measure.regionprops.
    It will return the standard deviation of the intensity of the image within
    the label of the region.
    """
    region_intensity_std = np.std(intensity_image[region_mask])
    return region_intensity_std


def intensity_median(region_mask: np.ndarray, intensity_image: np.ndarray) -> float:
    """This function is designed to be passed to the extra_properties argument
    of skimage.measure.regionprops.
    It will return the median of the intensity of the image within the label
    of the region.
    """
    region_intensity_median = np.median(intensity_image[region_mask])
    return region_intensity_median


def intensity_pct25(region_mask: np.ndarray, intensity_image: np.ndarray) -> float:
    """This function is designed to be passed to the extra_properties argument
    of skimage.measure.regionprops.
    It will return the 25th percentile of the intensity of the image within the
    label of the region.
    """
    region_intensity_pct25 = np.percentile(intensity_image[region_mask], q=25)
    return region_intensity_pct25


def intensity_pct75(region_mask: np.ndarray, intensity_image: np.ndarray) -> float:
    """This function is designed to be passed to the extra_properties argument
    of skimage.measure.regionprops.
    It will return the 75th percentile of the intensity of the image within the
    label of the region.
    """
    region_intensity_pct75 = np.percentile(intensity_image[region_mask], q=75)
    return region_intensity_pct75


def walk_the_line(
    skel: np.ndarray, max_num_pixels: int | None = None, bidirectional: bool = True
) -> tuple:
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
    skel: np.ndarray
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
            {line_first_coordinate:
                {next_closest_connected_coordinate: distance_to_closest_coordinate},
             ...,
             second_last_line_coordinate:
                {last_line_coordinate: distance_from_second_last_to_last_coordinate},
             last_line_coordinate: {}}
        Length of line1 and line2 should each be equal to the number of True pixels in skel.
        Returns (line1, line2) if bidirectional=True, else returns (line1,).
    """

    img_dim = skel.ndim
    max_num_pixels = max_num_pixels or np.count_nonzero(skel)

    coords = list(zip(*np.where(skel), strict=False))

    if len(coords) < 2:
        line1: dict = {(coords)[-1]: {}}
        if bidirectional:
            line2: dict = {(coords)[-1]: {}}
        else:
            pass
        pass

    else:
        coords1, coords2 = numpy_mesh_coords(coords, coords, indexing="xy")
        dists = np.linalg.norm(coords2 - coords1, axis=2)

        # conn1 = dists == 1
        # conn2 = dists == np.sqrt(2)
        # # conn3 = dists == np.sqrt(3)
        # conn_all = conn1 + conn2
        conns = [dists == np.sqrt(dim) for dim in range(1, img_dim + 1)]
        conn_all = sum(conns).astype(bool)

        ## now mask the array
        dists = np.ma.masked_array(data=dists, mask=dists == 0)

        edges_from_dist_ls = [
            np.all(
                dists[
                    (conn_all[i, :] * (conn_all[i, :] + conn_all))
                    * (conn_all[i, :] * (conn_all[i, :] + conn_all)).T
                ]
                > np.sqrt(img_dim)
            )
            for i in range(len(conn_all))
        ]

        edges_from_dist = np.array([x if x else False for x in edges_from_dist_ls])

        edge_conn = np.array([np.count_nonzero(conn_arr, axis=1) == 2 for conn_arr in conns])
        edge_anticonn = np.array([np.count_nonzero(conn_arr, axis=1) == 0 for conn_arr in conns])

        edges_from_conn = sum(
            [
                edge_conn[i, :]
                * ~(
                    sum([~edge_anticonn[j] for j in range(len(edge_anticonn)) if j != i]).astype(
                        bool
                    )
                )
                for i in range(len(conns))
            ]
        ).astype(bool)
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

        assert np.count_nonzero(nodes) == 2

        ## now that we know which coordinates are nodes or edges,
        ## we can spatially order them (i.e. walk from one node
        ## along the edge of the line to the other node)

        ## starting from one node, move to the closest neighbour
        ## create a dictionary of which coordinates are next to which

        conns_dict = {
            tuple(coords2[i, 0].tolist()): dict(
                zip(
                    [tuple(x) for x in coords1[i, conn_all[i]].tolist()],
                    dists[i, conn_all[i]].tolist(),
                    strict=False,
                )
            )
            for i in range(len(conn_all))
        }
        node_coords = [tuple(c) for c in np.asarray(coords)[nodes, :]]

        ## pick a starting position from one of the 2 endpoints of the coordinates
        ## (the choice is arbitrary)
        curr_node = node_coords[-1]
        visited_coords = [curr_node]
        line1 = {}

        for _ in range(max_num_pixels):
            line1[curr_node] = {
                n: conns_dict[curr_node][n]
                for n in conns_dict[curr_node]
                if conns_dict[curr_node][n]
                == min(
                    [
                        conns_dict[curr_node][k]
                        for k in conns_dict[curr_node].keys()
                        if k not in visited_coords
                    ],
                    default=[],
                )
                and n not in visited_coords
            }
            if line1[curr_node]:
                curr_node = tuple(line1[curr_node].keys())[-1]
                visited_coords.append(curr_node)
            else:
                break

        if bidirectional:
            curr_node = node_coords[-2]
            visited_coords = [curr_node]
            line2 = {}

            for _ in range(max_num_pixels):
                line2[curr_node] = {
                    n: conns_dict[curr_node][n]
                    for n in conns_dict[curr_node]
                    if conns_dict[curr_node][n]
                    == min(
                        [
                            conns_dict[curr_node][k]
                            for k in conns_dict[curr_node].keys()
                            if k not in visited_coords
                        ],
                        default=[],
                    )
                    and n not in visited_coords
                }
                if line2[curr_node]:
                    curr_node = tuple(line2[curr_node].keys())[-1]
                    visited_coords.append(curr_node)
                else:
                    break
        else:
            pass

    return (line1, line2) if bidirectional else (line1,)


def get_length(skel: np.ndarray, max_num_pixels: int | None = None) -> float:
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
    skel: np.ndarray
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
    (line,) = walk_the_line(skel, max_num_pixels, bidirectional=False)
    # get a list of the distances from one coordinate to the next
    dists = [
        line[startpoints][endpoints] for startpoints in line for endpoints in line[startpoints]
    ]
    # return the total of the distances as the length
    length = sum(dists)

    return length


def build_cdh5_measured_features_tables(
    dataset_name: str,
    tp: int,
    out_dir: str | Path,
    position: int = 0,
    save_output: bool | None = True,
    create_validation_image: bool = False,
) -> None:
    """
    Build tables of measured features from the segmentation images
    and the raw cdh5 images.
    The segmentation properties tables is
    a table of measured features extracted from the cdh5 segmentations
    using skimage.regionprops.
    The edge alignments table contains measured features that were
    determined based on a thresholded image of the cdh5 signal
    (i.e. they don't require the cdh5 segmentations).

    Also produces a validation image if requested
    (a validation image has segmentation borders, nodes, edges, and
    the straight lines connecting nodes as channels in a single
    .tiff image).

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to process.
    tp: int
        The timepoint to process.
    out_dir: str | Path
        The output directory to save the tables and validation images to.
    position: int
        The position to process (this will be equal to the scene index).
    save_output: bool | None
        Whether to save the output tables (and validation images if selected).
    create_validation_image: bool
        Whether to create a validation image.

    Returns
    -------
    This function will only save tables and images,
    it does not return anything.

    The tables contain the following information:
    segmentation properties table:
    - filepath_raw_image
    - filepath_segmentation_image
    - dataset_name
    - position
    - T
    - cell_label
    - cell_centroid
    - cell_area (px**2)
    - cell_perimeter (px)
    - cell_perimeter (px)
    - cell_solidity
    - major_axis_length
    - minor_axis_length
    - cell_eccentricity
    - cell_orientation
    - cell_fluorescence_mean (a.u.)
    - cell_fluorescence_std (a.u.)
    - cell_fluorescence_median (a.u.)
    - cell_fluorescence_min (a.u.)
    - cell_fluorescence_pct25 (a.u.)
    - cell_fluorescence_pct75 (a.u.)
    - cell_fluorescence_max (a.u.)
    - neighboring_cell_labels
    - edge_labels
    - node_labels
    - node_pair_labels
    - edge_fluorescences (a.u.)
    - node_fluorescences (a.u.)
    - touches_image_border
    - measurement_timestamp
    - git_branch_name
    - git_commit_hash
    - git_uncommitted_changes

    edge alignments table:
    - filepath_raw_image
    - filepath_raw_image
    - filepath_segmentation_image
    - dataset_name
    - position
    - T
    - node_pair_labels
    - node_pair_centroids
    - node_to_node_distance
    - angle_relative_to_horizontal
    - connecting_edges
    - edge_num_pixels
    - edge_length (px)
    - edge_fluorescence_mean (a.u.)
    - edge_fluorescence_std (a.u.)
    - edge_fluorescence_median (a.u.)
    - edge_fluorescence_min (a.u.)
    - edge_fluorescence_pct25 (a.u.)
    - edge_fluorescence_pct75 (a.u.)
    - edge_fluorescence_max (a.u.)
    - measurement_timestamp
    - git_branch_name
    - git_commit_hash
    - git_uncommitted_changes
    """

    import pandas as pd

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_image
    from endo_pipeline.library.process.general_image_preprocessing import save_image_output
    from endo_pipeline.manifests import (
        get_image_location_for_dataset,
        get_zarr_location_for_position,
        load_image_manifest,
    )
    from endo_pipeline.settings import DIMENSION_ORDER

    logger.debug(f"Working on {dataset_name} -- T={tp}...")

    dim_order = DIMENSION_ORDER

    out_dir = Path(out_dir)
    images_out_dir = out_dir / f"{dataset_name}/P{position}/images"
    tables_out_dir_alignments = out_dir / f"{dataset_name}/P{position}/tables_cdh5_alignments"
    tables_out_dir_segprops = (
        out_dir / f"{dataset_name}/P{position}/tables_cdh5_segmentation_properties"
    )

    logger.debug(f"T={tp} -- loading imaging datasets")

    # load the raw cdh5 image data
    dataset_config = load_dataset_config(dataset_name)
    image_loc = get_zarr_location_for_position(dataset_config, position)
    raw_arr = load_image(image_loc, channels=["EGFP"], timepoints=tp, level=0)
    raw_arr = raw_arr.max(axis=dim_order.index("Z")).squeeze().compute()
    voxel_size = load_image(image_loc, read=False).physical_pixel_sizes

    logger.debug(f"T={tp} -- loading classic segmentation")

    seg_manifest = load_image_manifest("cdh5_classic_seg_zarr")
    seg_location = get_image_location_for_dataset(seg_manifest, dataset_config, position)
    seg_arr = load_image(seg_location, squeeze=True, compute=True, timepoints=tp)
    seg_filepath = seg_location.path.as_posix() if seg_location.path is not None else ""

    # NOTE: the segmentation images are stored as a single channel and single timepoint
    seg_borders = segmentation.find_boundaries(seg_arr)

    ## convert cleaned up threshold of cadherin signal to nodes and edges
    logger.debug(f"T={tp} -- getting nodes and edges")
    nodes, edges, _, _ = arr2graph(seg_borders, closing_step=False)

    ## get the node-to-node distances and the angle between a line connecting two nodes
    ## and a horizontal line
    ## NOTE there should also be a way to get the error in the measurement of the angles too...
    logger.debug(f"T={tp} -- calculating distances and angles between neighboring nodes")

    neighbor_node_metrics, labeled_region_metrics = calculate_region_border_metrics(
        seg_borders.astype(bool), raw_arr, seg_arr
    )

    ## save a table of the results
    if save_output:
        tables_out_dir_alignments.mkdir(exist_ok=True, parents=True)
        ## save table output of edge alignments
        logger.debug(f"T={tp} -- saving table of edge angles and distances")
        table = pd.DataFrame(
            {
                "filepath_raw_image": image_loc.path.as_posix(),
                "filepath_segmentation_image": seg_filepath,
                "dataset_name": dataset_name,
                "position": position,
                "T": tp,
                "node_pair_labels": neighbor_node_metrics["node_pair_labels"],
                "node_pair_centroids": neighbor_node_metrics["node_pair_centroids"],
                "node_to_node_distance": neighbor_node_metrics["distances"],
                "angle_relative_to_horizontal": neighbor_node_metrics["angles"],
                "connecting_edges": neighbor_node_metrics["edge_labels"],
                "edge_num_pixels": neighbor_node_metrics["edge_num_pixels"],
                "edge_length (px)": neighbor_node_metrics["length (px)"],
                "edge_fluorescence_mean (a.u.)": neighbor_node_metrics["fluor_mean (au)"],
                "edge_fluorescence_std (a.u.)": neighbor_node_metrics["fluor_std (au)"],
                "edge_fluorescence_median (a.u.)": neighbor_node_metrics["fluor_median (au)"],
                "edge_fluorescence_min (a.u.)": neighbor_node_metrics["fluor_min (au)"],
                "edge_fluorescence_pct25 (a.u.)": neighbor_node_metrics["fluor_pct25 (au)"],
                "edge_fluorescence_pct75 (a.u.)": neighbor_node_metrics["fluor_pct75 (au)"],
                "edge_fluorescence_max (a.u.)": neighbor_node_metrics["fluor_max (au)"],
            }
        )
        table.to_parquet(
            tables_out_dir_alignments / f"{dataset_name}_P{position}_T{tp}_cdh5_alignments.parquet",
            index=False,
        )

        if create_validation_image:
            images_out_dir.mkdir(exist_ok=True, parents=True)
            ## save images containing the nodes, edges, and node-node lines
            ## as different channels
            logger.debug(f"T={tp} -- saving multichannel images of results for validation")

            ## create a rasterized image of the lines
            lines = np.zeros(nodes.shape, dtype=np.uint16)
            ## need to flatten node_coord_pairs first before passing to rasterize_edge_between_nodes
            node_coord_pairs = [
                node_coords
                for edge in neighbor_node_metrics["node_pair_centroids"]
                for node_coords in edge
            ]
            lines, _ = rasterize_edges_between_nodes(node_coord_pairs, lines, label_lines=True)

            ## organize the image data and save it
            out_path = images_out_dir / f"{dataset_name}_P{position}_T{tp}.ome.tiff"
            images_out = [seg_borders, nodes, edges, lines]
            images_out_metadata = {
                "image_name": dataset_name,
                "channel_names": ["segmentation_borders", "nodes", "edges", "lines"],
                "channel_colors": [
                    (255, 255, 255),
                    (255, 0, 255),
                    (0, 255, 255),
                    (255, 255, 0),
                ],
                "physical_pixel_sizes": voxel_size,
                "dim_order": "YX",
            }
            save_image_output(out_path, images_out, images_out_metadata)

        ## save table output of cell properties (e.g. areas, etc.)
        if labeled_region_metrics:
            tables_out_dir_segprops.mkdir(exist_ok=True, parents=True)
            logger.debug(f"T={tp} -- saving table of cell properties")
            table = pd.DataFrame(
                {
                    "filepath_raw_image": image_loc.path.as_posix(),
                    "filepath_segmentation_image": seg_filepath,
                    "dataset_name": dataset_name,
                    "position": position,
                    "T": tp,
                    "cell_label": labeled_region_metrics["cell_label"],
                    "cell_centroid": labeled_region_metrics["cell_centroid"],
                    "cell_area (px**2)": labeled_region_metrics["cell_area (px**2)"],
                    "cell_perimeter (px)": labeled_region_metrics["cell_perimeter (px)"],
                    "cell_solidity": labeled_region_metrics["cell_solidity"],
                    "major_axis_length": labeled_region_metrics["major_axis_length"],
                    "minor_axis_length": labeled_region_metrics["minor_axis_length"],
                    "cell_eccentricity": labeled_region_metrics["cell_eccentricity"],
                    "cell_orientation": labeled_region_metrics["cell_orientation"],
                    "cell_fluorescence_mean (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_mean (au)"
                    ],
                    "cell_fluorescence_std (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_std (au)"
                    ],
                    "cell_fluorescence_median (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_median (au)"
                    ],
                    "cell_fluorescence_min (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_min (au)"
                    ],
                    "cell_fluorescence_pct25 (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_pct25 (au)"
                    ],
                    "cell_fluorescence_pct75 (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_pct75 (au)"
                    ],
                    "cell_fluorescence_max (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_max (au)"
                    ],
                    "neighboring_cell_labels": labeled_region_metrics["neighboring_cell_labels"],
                    "edge_labels": labeled_region_metrics["edge_labels"],
                    "node_labels": labeled_region_metrics["node_labels"],
                    "node_pair_labels": labeled_region_metrics["node_pair_labels"],
                    "edge_fluorescences (a.u.)": labeled_region_metrics["edge_fluorescences (au)"],
                    "node_fluorescences (a.u.)": labeled_region_metrics["node_fluorescences (au)"],
                    "touches_image_border": labeled_region_metrics["touches_image_border"],
                }
            )
            table.to_parquet(
                tables_out_dir_segprops / f"{dataset_name}_P{position}_T{tp}_cdh5_segprops.parquet",
                index=False,
            )


def get_nuclei_features_from_image(
    cdh5_seg: np.ndarray | Array,
    nuc_seg: np.ndarray | Array,
    fluorescence_images: list[np.ndarray],
    fluor_img_names: list[str] | None = None,
    seg_dim_order: str = "YX",
) -> pd.DataFrame:
    """
    Extract features from nuclei segmentations and their overlap with cell segmentations.

    Parameters
    ----------
    cdh5_seg: ndarray
        Image of the cell segmentations based on Cdh5.
    nuc_seg: ndarray:
        Image of the nuclei segmentations.
    fluorescence_images: list[np.ndarray]:
        List of fluorescence images to get intensity information for each
        of the nuclei segmentation regions. In this workflow each image
        is a channel from the raw image.
    fluor_img_names: list[str] | None:
        Names of the fluorescence images. If None, defaults to "Channel_0", "Channel_1", etc.
    seg_dim_order: str:
        Order of dimensions that the segmentation images are in. Default is "YX".

    Returns
    -------
        pd.DataFrame: DataFrame with extracted features.
    """
    from skimage.measure import regionprops

    # just in case make sure that the number of dimensions provided
    # in seg_dim_order matches that of the images
    for img in [
        cdh5_seg,
        nuc_seg,
        *fluorescence_images,
    ]:
        assert len(seg_dim_order) == img.ndim

    # assign default names to fluorescence images if not provided
    channel_indices = range(len(fluorescence_images))
    if fluor_img_names is None:
        fluor_img_names = [f"Channel{i}" for i in channel_indices]

    # get intensities in the segmented nuclei regions
    # for each channel
    nuc_props_on_intens = {}
    for i in range(len(fluorescence_images)):
        nuc_props_on_intens[fluor_img_names[i]] = {
            prop.label: prop
            for prop in regionprops(label_image=nuc_seg, intensity_image=fluorescence_images[i])
        }

    nuc_seg_size_dict = {prop.label: int(prop.area) for prop in regionprops(nuc_seg)}

    # associate each nuclei with a cdh5 segmentation
    reg_props = regionprops(label_image=cdh5_seg, intensity_image=nuc_seg)

    # Set up some initial data containers to populate
    nuc_feats_ls: list = []

    feats_with_list_of_lists: dict[str, Callable] = {
        "nuc_seg_intens_means": np.mean,
        "nuc_seg_intens_stds": np.std,
        "nuc_seg_intens_medians": np.median,
        "nuc_seg_intens_pct25s": lambda x: np.percentile(x, 25),
        "nuc_seg_intens_pct75s": lambda x: np.percentile(x, 75),
        "nuc_seg_intens_maxs": np.max,
        "nuc_seg_intens_mins": np.min,
    }

    # Go through the region properties and extract features
    for prop in reg_props:
        nuc_seg_labels = np.unique(prop.intensity_image[prop.intensity_image != 0]).tolist()

        nuc_feats = {
            "cdh5_segmentation_label": prop.label,
            "nuclei_segmentation_labels": nuc_seg_labels,
            "nuclei_seg_in_cdh5_seg_frac": [],
        }

        for f in feats_with_list_of_lists.keys():
            [nuc_feats.update({f"{f}_{chan}": []}) for chan in fluor_img_names]

        # add the fraction overlap of the cdh5 segmentation with the segmentation
        # to each of the properties in reg_props
        # also add the label with the most overlap
        for lab in nuc_seg_labels:
            if nuc_seg_labels:
                nuc_seg_in_cdh5_seg_size = np.count_nonzero(prop.intensity_image == lab)
                nuc_seg_total_size = nuc_seg_size_dict[lab]
                nuc_feats["nuclei_seg_in_cdh5_seg_frac"].append(
                    nuc_seg_in_cdh5_seg_size / nuc_seg_total_size
                )

                # summarize intensities in segmented nuclei regions for each channel
                for chan in fluor_img_names:
                    nuc_arr = nuc_props_on_intens[chan][lab].image
                    intens_arr = nuc_props_on_intens[chan][lab].image_intensity

                    for feat, func in feats_with_list_of_lists.items():
                        nuc_feats[f"{feat}_{chan}"].append(func(intens_arr[nuc_arr]))

        nuc_lab_frac_dict = dict(
            zip(nuc_seg_labels, nuc_feats["nuclei_seg_in_cdh5_seg_frac"], strict=False)
        )
        nuclei_seg_with_most_overlap = [
            lab
            for lab in nuc_lab_frac_dict
            if nuc_lab_frac_dict[lab] == max(nuc_lab_frac_dict.values())
        ]
        for i, nuc_lab_max in enumerate(nuclei_seg_with_most_overlap):
            nuc_feats[f"nuclei_seg_with_most_overlap_{i}"] = nuc_lab_max
            for dim_index, dim in enumerate(seg_dim_order):
                nuc_feats[f"nuc_with_most_overlap_{i}_centroid_{dim}"] = float(
                    nuc_props_on_intens["BF"][nuc_lab_max].centroid[dim_index]
                )

        nuc_feats_ls.append(nuc_feats)

    nuc_feats_df = pd.DataFrame(nuc_feats_ls)

    return nuc_feats_df


def get_nuclei_features_from_dataset_at_timepoint(
    dataset_name: str,
    position: int,
    tp: int,
    out_dir: Path,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Load label-free nuclei prediction images and measure features for a given
    dataset, position, and timepoint.
    """

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_image
    from endo_pipeline.manifests import (
        get_image_location_for_dataset,
        get_zarr_location_for_position,
        load_image_manifest,
    )
    from endo_pipeline.settings import DIMENSION_ORDER

    # Load segmentations and image
    dim_order = DIMENSION_ORDER
    dataset_config = load_dataset_config(dataset_name)
    channel_names = dataset_config.channel_names

    nuc_manifest = load_image_manifest("nuclear_labelfree_seg_zarr")
    nuc_location = get_image_location_for_dataset(nuc_manifest, dataset_config, position)
    nuc_seg = load_image(nuc_location, squeeze=True, compute=True, timepoints=tp)

    cdh5_manifest = load_image_manifest("cdh5_classic_seg_zarr")
    cdh5_location = get_image_location_for_dataset(cdh5_manifest, dataset_config, position)
    cdh5_seg = load_image(cdh5_location, squeeze=True, compute=True, timepoints=tp)

    img_loc = get_zarr_location_for_position(dataset_config, position)
    raw_img = load_image(img_loc, channels=list(channel_names), timepoints=tp, level=0)
    raw_mip = raw_img.max(axis=dim_order.index("Z"), keepdims=True).compute()

    # split up the image into a list of channels
    channel_arrs = np.split(
        raw_mip, indices_or_sections=len(channel_names), axis=dim_order.index("C")
    )
    channel_arrs = [channel_arr.squeeze() for channel_arr in channel_arrs]

    # Get the nuclei properties
    nuc_feats_df = get_nuclei_features_from_image(
        cdh5_seg=cdh5_seg,
        nuc_seg=nuc_seg,
        fluorescence_images=channel_arrs,  # type:ignore[arg-type]
        fluor_img_names=channel_names,  # type:ignore[arg-type]
        seg_dim_order="YX",
    )

    # add the total number of detected nuclei per image to the dataframe
    num_nuclei = np.count_nonzero(np.unique(nuc_seg))
    nuc_feats_df["total_nuclei_count_at_T"] = num_nuclei

    # add the dataset name, position, and T to the dataframe
    nuc_feats_df["dataset_name"] = dataset_name
    nuc_feats_df["position"] = position
    nuc_feats_df["T"] = tp

    # move the dataset_name, position, and T columns to the front
    # of the data table
    nuc_feats_df = nuc_feats_df[
        ["dataset_name", "position", "T"]
        + [col for col in nuc_feats_df.columns if col not in ["dataset_name", "position", "T"]]
    ]

    if save_output:
        out_subdir = out_dir / dataset_name / f"P{position}"
        out_subdir.mkdir(exist_ok=True, parents=True)
        filename = f"{dataset_name}_P{position}_T{tp}_nuclei_labelfree_features.parquet"
        nuc_feats_df.to_parquet(out_subdir / filename, index=False)

    return nuc_feats_df


def build_cdh5_measured_features_tables_multiproc_wrapper(args: ImageProcessingArgs) -> None:
    """Build and save measured features tables using multiprocessing."""

    build_cdh5_measured_features_tables(
        out_dir=args.output_dir,
        dataset_name=args.dataset_name,
        tp=args.timepoint,
        position=args.position,
        save_output=args.save_output,
        create_validation_image=args.is_validation_image,
    )


def get_and_save_nuclei_features_arg_unpacker(args: ImageProcessingArgs) -> None:
    """Unpack arguments from an argument dictionary and call
    get_nuclei_features_from_dataset_at_timepoint.
    """

    get_nuclei_features_from_dataset_at_timepoint(
        out_dir=args.output_dir,
        dataset_name=args.dataset_name,
        tp=args.timepoint,
        position=args.position,
        save_output=args.save_output,
    )


def extract_t(
    fp_as_string: str | Path,
    int_only: bool = True,
    use_last_match: bool = True,
    default_if_not_found: int | str = "",
) -> str | int:
    """
    Extract the timepoint value from a string or Path.name.
    Searches for the pattern "T[0-9]+" to find the timepoint.
    If use_last_match is True then the last match will be used,
    otherwise the first one will be used.

    Parameters
    ----------
    fp_as_string: str or Path
        A string or Path.name to get the timepoint from.
    int_only: bool
        Whether to return just the timepoint as an integer or
        an entire string (i.e. 10 vs 'T010')
        Default is True.
    use_last_match: bool
        Whether to use the last match (in the event that multiple possible
        timepoint values were found in the string).
        If False then the first match will be used.
        E.g. image_name_T1_etc_T57.tif can return either T1 or T57, but
        will return 57 by default. Ideally the timepoint in fp_as_string
        would be unambiguous.
        Default is True.

    Returns
    -------
    t: int or str
        The timepoint represented as an integer if int_only is True, otherwise
        the timepoint represented as a string including the T before.
    """

    if isinstance(fp_as_string, Path):
        fp_as_string = str(fp_as_string)

    index = -1 if use_last_match else 0
    t = re.findall("T[0-9]+", fp_as_string)
    t_value = int(t[index].split("T")[-1]) if t else default_if_not_found
    if not t:
        logger.debug("""No 'T[0-9]+' found in filename. Using T == default_if_not_found.""")

    return t_value if int_only else f"T{t_value}"


def concatenate_and_save_feature_tables(
    out_dir: Path,
    dataset_name: str,
    out_file_suffix: str = "",
    input_filename_contains: str = "",
    file_extension: str = ".csv",
    sort_by_T: bool = True,
    check_saved_dataframe: bool = True,
    remove_initial_files_and_folders: bool = False,
) -> None:
    """
    Concatenate the nuclei feature tables for all positions and
    timepoints for a given dataset in an out_dir and then saves
    the concatenated table to the output directory.
    The expected file structure in out_dir is:
    out_dir/dataset_name/position/*filename_contains*.file_extension.
    """
    out_subdir = out_dir / dataset_name

    file_extension = f".{file_extension}" if not file_extension.startswith(".") else file_extension
    if input_filename_contains and not input_filename_contains.endswith("*"):
        input_filename_contains = f"{input_filename_contains}*"
    feats_filepaths = list(out_subdir.glob(f"**/*{input_filename_contains}{file_extension}"))
    if sort_by_T:
        feats_filepaths = sorted(feats_filepaths, key=lambda fp: extract_t(fp.stem))

    if file_extension == ".tsv":
        sep = "\t"
        table_reader = lambda fp: pd.read_csv(fp, sep=sep)
        table_writer = lambda df, fp: df.to_csv(fp, sep=sep, index=False)
    elif file_extension == ".csv":
        sep = ","
        table_reader = lambda fp: pd.read_csv(fp, sep=sep)
        table_writer = lambda df, fp: df.to_csv(fp, sep=sep, index=False)
    elif file_extension == ".parquet":
        table_reader = lambda fp: pd.read_parquet(fp)
        table_writer = lambda df, fp: df.to_parquet(fp, index=False)
    else:
        raise ValueError(
            f"Invalid file extension {file_extension}. Must be .csv, .tsv., or .parquet."
        )
    feats_dfs = [table_reader(fp) for fp in feats_filepaths]

    # define the output path for the concatenated dataframe
    if out_file_suffix:
        out_file_suffix = (
            f"_{out_file_suffix}" if not out_file_suffix.startswith("_") else f"{out_file_suffix}"
        )
    concatenated_df_out_path = out_dir / f"{dataset_name}{out_file_suffix}{file_extension}"

    if feats_dfs:
        concatenated_df = pd.concat(feats_dfs, ignore_index=True)
        table_writer(concatenated_df, concatenated_df_out_path)
    else:
        logger.debug(f"No feature tables found for {dataset_name}.")

    if check_saved_dataframe:
        # check that the concatenated dataframe at least has the same shape
        # and column names as a proxy for checking if it was saved correctly
        saved_df = table_reader(concatenated_df_out_path)
        same_shape = saved_df.shape == concatenated_df.shape
        same_column_names = all(saved_df.columns == concatenated_df.columns)
        if not (same_shape and same_column_names):
            raise ValueError(
                f"Saved dataframe {concatenated_df_out_path} \
                    does not match the concatenated dataframe."
            )
        logger.info(f"Concatenated dataframe saved to {concatenated_df_out_path}.")

    if remove_initial_files_and_folders:
        # remove files that match input_filename_contains
        for fp in feats_filepaths:
            fp.unlink()
    dirs_to_remove = list(out_subdir.glob("**/"))
    # remove the empty directory now that old tables are deleted
    # (note this must be done in reverse order because a folder with
    # subfolders does not count as empty and therfore raises an error)
    for dir_path in dirs_to_remove[::-1]:
        # NOTE that rmdir only removes empty directories
        # and will raise an error if it is not empty. If
        # a directory is not empty then we will skip it
        if not any(list(scandir(dir_path))):
            dir_path.rmdir()
            logger.debug(f"Removed empty directory {dir_path}.")
        else:
            logger.debug(f"Directory {dir_path} is not empty, skipping removal.")
            continue
