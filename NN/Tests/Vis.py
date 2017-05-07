from NN.Source.Basic.Networks import NNDist
from Util.Util import DataUtil
from Util.Timing import Timing


def vis_test():
    nn = NNDist()
    epoch = 1000
    record_period = 4

    timing = Timing(enabled=True)
    timing_level = 1
    x, y = DataUtil.gen_spiral(50, 3, 3, 2.5)
    nn.build([x.shape[1], 6, 6, 6, y.shape[1]])
    nn.optimizer = "Adam"
    nn.preview()
    nn.feed_timing(timing)
    nn.fit(x, y, verbose=1, record_period=record_period, epoch=epoch, train_only=True,
           draw_detailed_network=True, make_mp4=False, show_animation=True)
    nn.draw_results()
    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    vis_test()
