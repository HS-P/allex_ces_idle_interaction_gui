import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mars/allex_ces_idle_interaction/install/allex_ces_idle_interaction'
