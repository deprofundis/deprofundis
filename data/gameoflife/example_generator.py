from generator import Generator

import Tkinter, numpy

BURNIN = 2
ROWS = COLS = 20
OFFSET = 5
STEPS = 10

# Generate and initialize board
generator = Generator(ROWS, COLS)
board = generator.seed()

# plotting functionality
root = Tkinter.Tk()
canvas = [[0 for x in range(ROWS)] for y in range(COLS)]

# initializing canvase
for row in range(ROWS):
    for col in range(COLS):
        color = 'white' if board[row, col] is 0 else 'black'
        canvas[row][col] = Tkinter.Canvas(root, background=color, width=12, height=12, borderwidth=0)
        canvas[row][col].grid(row=row, column=col)

# initialize animation counter
count = 1


def animate():
    update()
    root.after(500, animate)


def update():
    board = generator.generate_step()
    print 'num ones' + str(numpy.sum(board)) + ' num zeros: ' + str(ROWS * COLS - numpy.sum(board))

    for row in range(ROWS):
        for col in range(COLS):
            color = 'white' if board[row, col] is 0 else 'black'
            canvas[row][col].configure(background=color)

animate()
root.mainloop()