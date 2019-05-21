import numpy as np
import pandas as pd
from scipy.spatial import distance
import pytesseract
from pytesseract import Output
# import cv2

dict_of_elements = {'Type': ['Dc_voltage', 'Resistor', 'Inductor', 'Capacitor', 'Resistor'],
                    'Coordinate': [(0.3, 0.1), (0.1, 0.3), (0.1, 0.8), (0.3, 0.5), (0.3, 0.9)]}
# (0.3, 0.1): 'Dc_voltage', (0.1, 0.3): 'Resistor',
#                     (0.1, 0.8): 'Inductor', (0.3, 0.5): 'Capacitor',
#                     (0.3, 0.9): 'Resistor'
# df_of_elements = pd.DataFrame(data=dict_of_elements)

dict_of_component_details = {'Label': ['R1', 'H1', 'R2', 'V1', 'C1'],
                             'Value': ['10K', '0.5H', '3K', '10V', '0.65uF'],
                             'Coordinate': [(0.05, 0.3), (0.051, 0.8), (0.3, 0.99), (0.3, 0.05), (0.3, 0.65)]}
# 'V1': '10V', 'R1': '10K', 'H1': '0.5H', 'C1': '0.65uF', 'R2': '3K'
# df_of_component_details = pd.DataFrame(data=dict_of_component_details)

dict_of_nodes = {'Coordinate': [(0.35, 0.5), (0.32, 0.1), (0.35, 0.9),
                                (0.1, 0.2), (0.25, 0.2),
                                (0.1, 0.42), (0.1, 0.75), (0.25, 0.5),
                                (0.1, 0.92), (0.25, 0.9)],
                 'Node': [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]}


# (0.35, 0.5): 0, (0.32, 0.1): 0, (0.35, 0.9): 0,
#                  (0.1, 0.2): 1, (0.25, 0.2): 1,
#                  (0.1, 0.42): 2, (0.1, 0.75): 2, (0.25, 0.5): 2,
#                  (0.1, 0.92): 3, (0.25, 0.9): 3

# 0: [(0.35, 0.5), (0.32, 0.1), (0.35, 0.9)],
#                  1: [(0.1, 0.2), (0.25, 0.2)],
#                  2: [(0.1, 0.42), (0.1, 0.75), (0.25, 0.5)],
#                  3: [(0.1, 0.92), (0.25, 0.9)]

# df_of_nodes = pd.DataFrame(data=list(dict_of_nodes.items()), columns=['Coordinate', 'Node'])


def center_comparison_item_matcher(dict_of_elements, dict_of_component_details, dict_of_nodes):
    # comparing center coordinates of elements and component details and
    # reordering the component details dictionary in the same way as the order of the elements dictionary.

    dist_element_detail = distance.cdist(list(dict_of_elements['Coordinate']),
                                         list(dict_of_component_details['Coordinate']), metric='euclidean')
    temp_order = dist_element_detail.argmin(axis=1)

    dict_of_component_details['Label'] = list(np.array(dict_of_component_details['Label'])[temp_order])
    dict_of_component_details['Value'] = list(np.array(dict_of_component_details['Value'])[temp_order])
    dict_of_component_details['Coordinate'] = list(np.array(dict_of_component_details['Coordinate'])[temp_order])

    # comparing center coordinates of elements and nodes and
    # reordering the nodes dictionary in the same way as the order of the elements dictionary.

    dist_element_node = distance.cdist(list(dict_of_elements['Coordinate']), list(dict_of_nodes['Coordinate']),
                                       metric='euclidean')
    temp_node_list = []
    [temp_node_list.append(list(i[:2])) for i in dist_element_node.argsort(axis=1)]
    temp_node_list = np.array(temp_node_list)  # .reshape(1, -1)

    dict_of_nodes['Coordinate'] = list(np.array(dict_of_nodes['Coordinate'])[temp_node_list])
    dict_of_nodes['Node'] = list(np.array(dict_of_nodes['Node'])[temp_node_list])

    dit = {**dict_of_elements, **dict_of_component_details, **dict_of_nodes}
    df_of_output = pd.DataFrame(data=dit)
    _ = df_of_output.pop('Coordinate')
    print(dict_of_elements, '\n', dict_of_component_details, '\n', dict_of_nodes)

    return df_of_output


# function for creating a dictionary with center coordinate and type of detected objects. (Y,X) COORDINATES ARE USED.
# SAMPLE INPUT ==> box_class={(0.23678646981716156,0.2858278155326843,0.3105493187904358,0.46136170625686646): [
# 'resistor: 60%'],(0.4042844772338867,0.7420671582221985,0.6319484710693359,0.8116713166236877): ['resistor: 63%']}
# SAMPLE OUTPUT ==> obj_dict={(0.2736678943037987, 0.3735947608947754): 'resistor', (0.5181164741516113,
# 0.7768692374229431): 'resistor'}
def obj_type_center_coord(box_class):
    temp_coord = np.array(list(box_class.keys()))
    temp_type = np.array(list(box_class.values()))

    obj_coord_center = np.array([(temp_coord[:, 0] + (temp_coord[:, 2] - temp_coord[:, 0]) / 2),
                                 (temp_coord[:, 1] + (temp_coord[:, 3] - temp_coord[:, 1]) / 2)]).T
    obj_type = np.array((list(np.char.split(temp_type.reshape((temp_type.shape[0],)), sep=':'))))[:, 0]

    obj_dict = {}
    for key, value in zip(obj_coord_center, obj_type):
        obj_dict[tuple(key)] = value

    return obj_dict


def pre_process_and_run_ocr(image_path):
    """
        process the image before it's passed to the tesseract ocr inorder to enhance the detection accuracy

    :param image_path: string of the image's path.
    :return: img: processed image that is ready for ocr to be applied.
    """

    img = cv2.imread(image_path)

    # Rescale the image, if needed.
    # img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Apply blur to smooth out the edges
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Thresholding types
    # img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    tesser_output = pytesseract.image_to_data(img, lang='Greek+eng', config='--psm 12', output_type=Output.DICT)

    return tesser_output


# function for creating a dictionary with the detected text and its bounding box center coordinate. (Y,X) COORDINATES
# ARE USED. SAMPLE INPUT ==> tesser_output={'level': [1, 2, 3, 4, 5, 5, 5, 4, 5], 'page_num': [1, 1, 1, 1, 1, 1, 1,
# 1, 1], 'block_num': [0, 1, 1, 1, 1, 1, 1, 1, 1], 'par_num': [0, 0, 1, 1, 1, 1, 1, 1, 1], 'line_num': [0, 0, 0, 1,
# 1, 1, 1, 2, 2], 'word_num': [0, 0, 0, 0, 1, 2, 3, 0, 1], 'left': [0, 4, 4, 5, 5, 44, 84, 4, 4], 'top': [0, 5, 5, 5,
# 5, 5, 5, 23, 23], 'width': [153, 142, 142, 141, 30, 30, 62, 30, 30], 'height': [41, 28, 28, 13, 13, 10, 13, 10,
# 10], 'conf': ['-1', '-1', '-1', '-1', 96, 93, 88, '-1', 42], 'text': ['', '', '', '', 'Long', 'live', 'Ethiopia',
# '', '1880']} , image_height_width=(41,153)
# SAMPLE OUTPUT ==> ocr_dict={(0.2804878048780488, 0.13071895424836602):
# 'Long', (0.24390243902439024, 0.38562091503267976): 'live', (0.2804878048780488, 0.7516339869281046): 'Ethiopia',
# (0.6829268292682927, 0.12418300653594772): '1880'}
def ocr_text_center_coord(tesser_output, image_height_width):
    img_height, img_width = image_height_width
    num_boxes = len(tesser_output['level'])
    temp_coord = []
    temp_text = []
    ocr_dict = {}

    for i in range(num_boxes):
        if tesser_output['text'][i].strip() != "":
            (x, y, width, height, text) = (
                tesser_output['left'][i], tesser_output['top'][i], tesser_output['width'][i],
                tesser_output['height'][i],
                tesser_output['text'][i])
            temp_coord.append([y, x, y + height, x + width])
            temp_text.append(text)

    temp_coord = np.array(temp_coord)
    temp_coord_center = np.array([(temp_coord[:, 0] + (temp_coord[:, 2] - temp_coord[:, 0]) / 2) / img_height,
                                  (temp_coord[:, 1] + (temp_coord[:, 3] - temp_coord[:, 1]) / 2) / img_width]).T
    for key, value in zip(temp_coord_center, temp_text):
        ocr_dict[tuple(key)] = value

    return ocr_dict


def generate_cir(df_of_elements, title='Circuit definition'):
    temp_file = open('temp_ng_spice.cir', 'w')
    temp_file.writelines([title, '\n'])

    # component definition
    for row in df_of_elements.itertuples(index=False):
        element_type, label, value, node = row
        start_node = node[0]
        end_node = node[1]
        node = str(start_node) + ' ' + str(end_node)
        temp_file.writelines([label, '\t', node, '\t', value, '\n'])

    # Footer definition
    closing1 = '.op'
    closing2 = '.end'
    footer = [closing1, '\n', closing2]

    # Write footer to file
    temp_file.writelines(footer)
    temp_file.close()


class ElectricalElmnt:
    # make them unaccessable from outside

    default_start_node = 0
    default_end_node = 1

    def __init__(self, center_coord=None, elmnt_type=None, elmnt_value=None, elmnt_label=None,
                 node_pair=(default_start_node, default_end_node)):
        self.center_coord = center_coord
        self.elmnt_type = elmnt_type
        self.elmnt_value = elmnt_value
        self.elmnt_label = elmnt_label
        self.node_pair = node_pair


class ElectricalCkt:
    # function to generate netlist and export the netlist to python commands which
    # will be later used to redraw the circuit diagram using the scheme draw python package
    def export_to_schem_draw(self):
        # TODO
        pass

    # function to generate and export netlist of the circuit to a cir file which will be used
    # by ngspice for simulation
    def export_to_cir(self):
        # TODO
        pass

    # function to redraw and display the circuit diagram usng the scheme draw python package
    def draw_ckt(self):
        # TODO
        pass

    # function to start simulation on the circuit using ngspice
    def simulate(self):
        # TODO
        pass


__end__ = '__end__'

if __name__ == '__main__':
    # test center_comparison_item_matcher() function
    df = center_comparison_item_matcher(dict_of_elements=dict_of_elements,
                                        dict_of_component_details=dict_of_component_details,
                                        dict_of_nodes=dict_of_nodes)

    # test generate_cir() function
    generate_cir(df, title='Circuit with manual inputs')
    pass
