import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yeah2/allex_ces_idle_interaction_gui/install/allex_ces_idle_interaction'
