from generator import Generator

import time, plotting, numpy

BURNIN = 2
ROWS = COLS = 20
OFFSET = 5
STEPS = 10

# Generate and initialize board
generator = Generator(ROWS, COLS)
board = generator.seed(offset=OFFSET)

for step in range(STEPS):
    board = generator.generate_step()
    plotting.plot_rbm_2layer(v_plus=board,fignum=1)

    print 'num ones' + str(numpy.sum(board)) + ' num zeros: ' + str(ROWS * COLS - numpy.sum(board))
    time.sleep(5.0)