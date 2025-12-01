import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dgx_allex_one/allex_ces_idle_interaction_gui/install/camera_node'
