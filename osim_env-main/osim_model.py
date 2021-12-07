"""
Functions to modify muscles and joints of osim model
"""

def lock_Coord(osim_file, coords, lock):
    """
    Lock or unlock some coordinates
    INPUTS: - osim_file: string, path to osim model
            - coords: string array, list of coordinates to lock or unlock
            - lock: bool, to lock or unlock the previous coordinates
    OUTPUT: - osim_file: string, path to modified osim model
    """
    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Coordinate':
            if line.split()[1].split('"')[1] in coords:
                new_lines[l+10] = '\t\t\t\t\t\t\t<locked>'+lock+'</locked>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file


def modify_default_Coord(osim_file, coord, value):
    """
    Modify the default value of a coordinate
    INPUTS: - osim_file: string, path to osim model
            - coord: string, coordinate to modify
            - value: float, new default value
    OUTPUT: - osim_file: string, path to modified osim model
    """
    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Coordinate':
            if line.split()[1].split('"')[1] == coord:
                new_lines[l + 2] = '\t\t\t\t\t\t\t<default_value>'+str(value)+'</default_value>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file

