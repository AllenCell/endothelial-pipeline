from bioio import BioImage
from pathlib import Path
import numpy as np

def get_nested_keys(nested_dict, ls=[], iterable_size_limit=50, check_for_lists=False):
    '''
    This function will return all the keys in a nested dictionary. It is a generator function.
    The keys are returned similar to globbing through paths in a file system, where the keys
    of a nested dictionary are returned as a list for each terminal key.
    `iterable_size_limit` sets an upper limit to the number of items  that can be subheading
    of the metadata and if this limit is exceeded then it will be skipped.
    `check_for_lists` lets you iterate through lists of dicionaries if they are present.
    The .sldy files from the 3i microscope have lists of dictionaries in the metadata.

    IMPORTANT!: WHEN CALLING THIS FUNCTION YOU MUST EXPLICITLY set ls=[], OTHERWISE IT WILL
    RETAIN INFORMATION FROM THE PREVIOUS CALLS TO THIS FUNCTION.

    Example:
    >>> nested_dict = {'a': {'b': {'c1': {'d1': 1, 'd2': 2, 'd3': 3},
                               'c2': {'e1': 1, 'e2': 2, 'e3': 3},},},
                       'aa': {'bb': {'cc': {'dd1': 1, 'dd2': 2, 'dd3': 3}}},
                       'aaa': {'bbb': {'ccc': {'ddd1': 1, 'ddd2': 2, 'ddd3': 3}}}}
    >>> keys = [x for x in get_nested_keys(nested_dict, ls=[])]
    >>> keys
    [['a', 'b', 'c1', 'd1'],
     ['a', 'b', 'c1', 'd2'],
     ['a', 'b', 'c1', 'd3'],
     ['a', 'b', 'c2', 'e1'],
     ['a', 'b', 'c2', 'e2'],
     ['a', 'b', 'c2', 'e3'],
     ['aa', 'bb', 'cc', 'dd1'],
     ['aa', 'bb', 'cc', 'dd2'],
     ['aa', 'bb', 'cc', 'dd3'],
     ['aaa', 'bbb', 'ccc', 'ddd1'],
     ['aaa', 'bbb', 'ccc', 'ddd2'],
     ['aaa', 'bbb', 'ccc', 'ddd3']]
    '''

    for key, val in nested_dict.items():
        # make a copy of the list of keys from the previous iteration
        ls_past = ls.copy()
        # add the current key to the list of keys
        ls.append(key)
        # print where you are in the nested dictionary
        if isinstance(val, dict):
            # if the value is a dictionary, recursively call this function
            # using the value as the new dictionary argument and the current
            # list as the list of keys to be added to
            if len(val) <= iterable_size_limit:
                yield from get_nested_keys(val, ls, iterable_size_limit, check_for_lists)
            else:
                ls.append(f'Over {iterable_size_limit} items. Skipping...')
                yield ls
        elif check_for_lists and isinstance(val, list):
        # elif isinstance(val, list):
            # if the value is a list, then convert it to a dictionary with
            # the indices as the keys and then call this function recursively
            if any(isinstance(x, dict) for x in val):
                val = {k:v for k,v in zip(range(len(val)), val) if isinstance(v, dict)}
                if len(val) <= iterable_size_limit:
                    yield from get_nested_keys(val, ls, iterable_size_limit, check_for_lists)
                else:
                    ls.append(f'Over {iterable_size_limit} items. Skipping...')
                    yield ls
            else:
                # if val has no dictionaries in it, then return the list of keys
                yield ls
        else:
            # if the value is not a dictionary, return the list of keys
            yield ls
        # if you've made it this far then that means that you've reached the
        # end of the nested dictionary, and so you need to go back up one or
        # more levels in the nested dictionary. Therefore reset the list of
        # keys to that previous iteration
        ls = ls_past.copy()


def get_sldy_metadata(filepath: Path):# -> dict:
    """Returns the metadata from a .sldy file which is a series of nested dictionaries."""
    img = BioImage(filepath)
    return img.metadata

def get_voxel_size(sldy_metadata: dict) -> dict:
    """Returns the voxel size in microns for each dimension from the output of the get_sldy_metadata function."""
    # below is the xy pixel size
    pixel_sizes_xy = sldy_metadata['image_record']['CLensDef70']['mMicronPerPixel']
    # NOTE that our 3i microscope has a piece of hardware that adjusts the magnification
    # called an optovar. This piece of hardware may not be present in other microscopes.
    # I am unsure of if the `['image_record']['COptovarDef70']['mMagnification']` field
    # would be missing if there was no optovar.
    optovar_mag = sldy_metadata['image_record']['COptovarDef70']['mMagnification']
    pixel_size_xy = pixel_sizes_xy / optovar_mag
    # below is the Z-step size
    pixel_size_z = sldy_metadata['channel_record']['CExposureRecord70'][0]['mInterplaneSpacing']
    voxel_size = {'X': pixel_size_xy, 'Y': pixel_size_xy, 'Z': pixel_size_z}
    return voxel_size

def get_objective_info(sldy_metadata: dict) -> dict:
    """Returns information about the objective used (the magnification and numerical aperture) from the output of the get_sldy_metadata function."""
    objective_info = {'magnification': sldy_metadata['image_record']['CLensDef70']['mActualMagnification'],
                      'numerical_aperture': sldy_metadata['image_record']['CLensDef70']['mNA']}
    return objective_info

def get_num_unique_imaging_positions(sldy_metadata: dict) -> int:
    """Returns the number of unique imaging positions from the output of the get_sldy_metadata function."""
    stage_position_dim_order = {dim: i for i, dim in enumerate(sldy_metadata['stage_position_data']['StructDefMemberName'])}
    stage_position_data = sldy_metadata['stage_position_data']['StructArrayValues']
    stage_position_data = np.reshape(stage_position_data, (-1, len(stage_position_dim_order)))
    num_horizontal_tiles = np.unique(stage_position_data[:,'mX']).shape
    num_vertical_tiles = np.unique(stage_position_data[:,'mY']).shape
    num_planes = sldy_metadata['image_record']['CImageRecord70']['mNumPlanes']
    return num_horizontal_tiles * num_vertical_tiles * num_planes

def get_num_timepoints(sldy_metadata: dict) -> int:
    """Returns the number of timepoints in the dataset from the output of the get_sldy_metadata function."""
    # NOTE: the `mNumTimepoints` field counts each acquisition as a timepoint.
    # Therefore if you acquired a multi-position or tiled timelapse, then the
    # number returned by `mNumTimepoints` will not be the number of timepoints
    # that was chosen when acquiring the imaging data. Therefore you must
    # determine the number of positions acquired first before determining the
    # number of timepoints.
    # E.g. a 3x3 tiled timelapse with 10 timepoints would have "90" under the
    # `mNumTimepoints` field.
    stage_position_dim_order = {dim: i for i, dim in enumerate(sldy_metadata['stage_position_data']['StructDefMemberName'])}
    stage_position_data = sldy_metadata['stage_position_data']['StructArrayValues']
    stage_position_data = np.reshape(stage_position_data, (-1, len(stage_position_dim_order)))
    num_horizontal_tiles = np.unique(stage_position_data[:,stage_position_dim_order['mX']]).size
    num_vertical_tiles = np.unique(stage_position_data[:,stage_position_dim_order['mY']]).size
    num_positions_acquired = sldy_metadata['image_record']['CImageRecord70']['mNumTimepoints']
    num_timepoints = num_positions_acquired / (num_horizontal_tiles * num_vertical_tiles)
    return round(num_timepoints, ndigits=0)

def get_tiling_percentage_overlap(sldy_metadata: dict) -> dict:
    """Returns the percentage overlap in X and Y for the tiling from the output of the get_sldy_metadata function."""
    # the width and height are recorded in pixel units
    fov_width = sldy_metadata['image_record']['CImageRecord70']['mWidth']
    fov_height = sldy_metadata['image_record']['CImageRecord70']['mHeight']

    # the dimension order of the stage position data is below
    stage_position_dim_order = {dim: i for i, dim in enumerate(sldy_metadata['stage_position_data']['StructDefMemberName'])}
    # the stage position data is saved as a 1D array, so we will reshape it
    # using the dimension order above
    # note that the stage position data is kept in physical units (i.e. microns)
    stage_position_data = sldy_metadata['stage_position_data']['StructArrayValues'][:18]
    stage_position_data = np.reshape(stage_position_data, (-1, len(stage_position_dim_order)))

    # next we will calculate the amount that the FOV shifts in X and Y
    # between each acquisition. This should let us figure out the overlap.
    # note that these are also in physical units.
    stage_increments_x = np.diff(stage_position_data[:,stage_position_dim_order['mX']])
    stage_increments_y = np.diff(stage_position_data[:,stage_position_dim_order['mY']])

    # we need the pixel resolution to conver the fov_width and fov_height to
    # physical units so that we can combine it with the stage increments to
    # calculate the percentage overlap
    px_res = sldy_metadata['image_record']['CLensDef70']['mMicronPerPixel'] / sldy_metadata['image_record']['COptovarDef70']['mMagnification']

    # conver the width and height to physical units
    fov_width_physical_size = fov_width * px_res
    fov_height_physical_size = fov_height * px_res

    # calculate the percentage overlap:
    percent_overlap_x = np.unique(np.round(100 * (1 - abs(stage_increments_x / fov_width_physical_size)), decimals=1))
    percent_overlap_y = np.unique(np.round(100 * (1 - abs(stage_increments_y / fov_height_physical_size)), decimals=1))
    return {'overlap_in_X': int(*percent_overlap_x), 'overlap_in_Y': int(*percent_overlap_y)}

def get_tiling_arrangement(sldy_metadata: dict) -> dict:
    """Returns the number of tiles in X and Y from the output of the get_sldy_metadata function."""
    stage_position_dim_order = {dim: i for i, dim in enumerate(sldy_metadata['stage_position_data']['StructDefMemberName'])}
    stage_position_data = sldy_metadata['stage_position_data']['StructArrayValues']
    stage_position_data = np.reshape(stage_position_data, (-1, len(stage_position_dim_order)))
    num_horizontal_tiles = np.unique(stage_position_data[:,stage_position_dim_order['mX']]).shape
    num_vertical_tiles = np.unique(stage_position_data[:,stage_position_dim_order['mY']]).shape
    # num_planes = sldy_metadata['image_record']['CImageRecord70']['mNumPlanes']
    return {'number_of_tiles_in_X': num_horizontal_tiles, 'number_of_tiles_in_Y': num_vertical_tiles}

def get_imaging_date(sldy_metadata: dict) -> dict:
    """
    Returns the date that the data was saved from the output of the get_sldy_metadata function.
    NOTE: This may not be the same date that imaging began!
    """
    # I considered returning the date as a datetime object but opted for a dictionary for
    # consistency with the other functions.
    imaging_date = {'year': sldy_metadata['image_record']['CImageRecord70']['mYear'],
                    'month': sldy_metadata['image_record']['CImageRecord70']['mMonth'],
                    'day': sldy_metadata['image_record']['CImageRecord70']['mDay'],
                    'hour': sldy_metadata['image_record']['CImageRecord70']['mHour'],
                    'minute': sldy_metadata['image_record']['CImageRecord70']['mMinute'],
                    'second': sldy_metadata['image_record']['CImageRecord70']['mSecond']}
    return imaging_date


# TODO: find out how to tell if a channel has fluorescence or brightfield and adjust excitation and emission accordingly
def get_channel_name(sldy_metadata: dict, return_unprocessed_string=False) -> list:
    """
    Returns the name of each channel from the output of the get_sldy_metadata function.
    NOTE: Channel names may need further processing to remove extraneous characters if
    `return_unprocessed_string = True`."""
    channel_names = [sldy_metadata['channel_record']['CFluorDef70'][i]['mName'] for i in range(sldy_metadata['image_record']['CImageRecord70']['mNumChannels'])]
    channel_names = channel_names if return_unprocessed_string else [x.split('_#32;')[0] for x in channel_names]
    return channel_names

def get_excitation_wavelength(sldy_metadata: dict) -> dict[str, float]:
    """
    Returns the excitation wavelength in nanometers for each channel from the output of the get_sldy_metadata function.
    CAUTION: IF ONE OF YOUR CHANNELS IS BRIGHTFIELD THEN THE OUTPUT FOR THAT CHANNEL MAY NOT BE CORRECT.
    """
    channel_names = [sldy_metadata['channel_record']['CFluorDef70'][i]['mName'] for i in range(sldy_metadata['image_record']['CImageRecord70']['mNumChannels'])]
    channel_names = [x.split('_#32;')[0] for x in channel_names]
    excitation_wavelengths = {chan: float(sldy_metadata['channel_record']['CFluorDef70'][i]['mExcitationLambda']) for i, chan in enumerate(channel_names)}
    return excitation_wavelengths

def get_emission_wavelength(sldy_metadata: dict) -> dict[str, float]:
    """
    Returns the emission wavelength in nanometers for each channel from the output of the get_sldy_metadata function.
    CAUTION: IF ONE OF YOUR CHANNELS IS BRIGHTFIELD THEN THE OUTPUT FOR THAT CHANNEL MAY NOT BE CORRECT.
    """
    channel_names = [sldy_metadata['channel_record']['CFluorDef70'][i]['mName'] for i in range(sldy_metadata['image_record']['CImageRecord70']['mNumChannels'])]
    channel_names = [x.split('_#32;')[0] for x in channel_names]
    emission_wavelengths = {chan: float(sldy_metadata['channel_record']['CFluorDef70'][i]['mLambda']) for i, chan in enumerate(channel_names)}
    return emission_wavelengths

def get_exposure_time(sldy_metadata: dict) -> dict[str, int]:
    """
    Returns the exposure time for each channel in milliseconds from the output of the get_sldy_metadata function.
    CAUTION: IF ONE OF YOUR CHANNELS IS BRIGHTFIELD THEN THE OUTPUT FOR THAT CHANNEL MAY NOT BE CORRECT.
    """
    channel_names = [sldy_metadata['channel_record']['CFluorDef70'][i]['mName'] for i in range(sldy_metadata['image_record']['CImageRecord70']['mNumChannels'])]
    channel_names = [x.split('_#32;')[0] for x in channel_names]
    exposure_times = {chan: int(sldy_metadata['channel_record']['CExposureRecord70'][i]['mExposureTime']) for i, chan in enumerate(channel_names)}
    return exposure_times

def get_time_intervals(sldy_metadata: dict, units='msec') -> dict[str, float]:
    """Returned time interval for each channel (default is in milliseconds) from the output of the get_sldy_metadata function.
    Possible options for 'units' argument are:
        'msec': milliseconds
        'sec': seconds
        'min': minutes
        'hr': hours
    CAUTION: IF ONE OF YOUR CHANNELS IS BRIGHTFIELD THEN THE OUTPUT FOR THAT CHANNEL MAY NOT BE CORRECT.
    """
    channel_names = [sldy_metadata['channel_record']['CFluorDef70'][i]['mName'] for i in range(sldy_metadata['image_record']['CImageRecord70']['mNumChannels'])]
    channel_names = [x.split('_#32;')[0] for x in channel_names]
    conversion_factors = {'msec': 1, 'sec': 1000, 'min': 60*1000, 'hr': 60*60*1000}
    time_intervals = {chan: sldy_metadata['channel_record']['CExposureRecord70'][i]['mTimeLapseInterval'] for i, chan in enumerate(channel_names)}
    time_intervals = {chan: float(t / conversion_factors[units]) for chan, t in time_intervals.items()}
    return time_intervals


def get_test_of_metadata():
    md_test = {'a': {'b': {'c1': {'d1': 1, 'd2': 2, 'd3': 3},
                        'c2': {'e1': 1, 'e2': 2, 'e3': 3},},},
            'aa': {'bb': {'cc': {'dd1': 1, 'dd2': 2, 'dd3': 3}}},
            'aaa': {'bbb': {'ccc': {'ddd1': 1, 'ddd2': 2, 'ddd3': 3}}}}
    md_keys = [x for x in get_nested_keys(md_test, ls=[], iterable_size_limit=10, check_for_lists=True)]
    # To see the keys in the nested dictionary:
    # [x for x in md_keys]
    return md_test, md_keys

def get_example_metadata():
    image_path = Path("//allen/aics/microscopy/Endo Timelapses/20241120/20241120_20X_timelapse_SLDY.dir")
    metadata = get_sldy_metadata(image_path)
    return metadata


# Example usage:
def show_example_usage():
    metadata = get_example_metadata()
    print('What are the keys / headers in the metadata?')
    print([x for x in get_nested_keys(metadata, ls=[], iterable_size_limit=50, check_for_lists=True)])

    print('What is the magnification of the objective used to collect these images?')
    print(get_objective_info(metadata)['magnification'])

    print('What are the channel names for this acquisition?')
    print(get_channel_name(metadata))

    print('What is the time interval between each acquisition in minutes? seconds?')
    print(f"{get_time_intervals(metadata, units='min')} minutes or {get_time_intervals(metadata, units='sec')} seconds.")

    print('What are is the voxel size for this acquisition?')
    print(get_voxel_size(metadata))
