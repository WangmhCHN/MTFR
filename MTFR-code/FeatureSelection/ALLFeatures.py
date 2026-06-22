'''
    保留所有变量
'''
import sys
sys.path.append(r"D:\WangMH_SYSU\PyProject\Project\MTFR2")


def Feature_Selection(data, target, args):
    Features_sorted = list(data.columns)
    figs = ()
    Importance_df = None
    return Features_sorted, figs, Importance_df
