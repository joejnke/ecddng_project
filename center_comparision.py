import numpy as np
import pandas as pd
from scipy.spatial import distance

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

    dist_element_detail = distance.cdist(list(dict_of_elements['Coordinate']), list(dict_of_component_details['Coordinate']), metric='euclidean')
    temp_order = dist_element_detail.argmin(axis=1)

    dict_of_component_details['Label'] = list(np.array(dict_of_component_details['Label'])[temp_order])
    dict_of_component_details['Value'] = list(np.array(dict_of_component_details['Value'])[temp_order])
    dict_of_component_details['Coordinate'] = list(np.array(dict_of_component_details['Coordinate'])[temp_order])

    # comparing center coordinates of elements and nodes and
    # reordering the nodes dictionary in the same way as the order of the elements dictionary.

    dist_element_node = distance.cdist(list(dict_of_elements['Coordinate']), list(dict_of_nodes['Coordinate']), metric='euclidean')
    temp_node_list = []
    [temp_node_list.append(list(i[:2])) for i in dist_element_node.argsort(axis=1)]
    temp_node_list = np.array(temp_node_list)   #.reshape(1, -1)

    dict_of_nodes['Coordinate'] = list(np.array(dict_of_nodes['Coordinate'])[temp_node_list])
    dict_of_nodes['Node'] = list(np.array(dict_of_nodes['Node'])[temp_node_list])

    print(dict_of_elements, '\n', dict_of_component_details, '\n', dict_of_nodes)

    #return df_of_output


__end__ = '__end__'

if __name__ == '__main__':
    print(center_comparison_item_matcher(dict_of_elements=dict_of_elements,
                                         dict_of_component_details=dict_of_component_details,
                                         dict_of_nodes=dict_of_nodes))

    pass

