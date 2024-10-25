import random
import numpy as np
import math
import statistics
import time
import pandas as pd
import os
import copy

df = pd.DataFrame(columns=['Filename', 'Difficulty', 'Clues', 'Empty', 'Execution Time (s)', 'Solved'])

def open_board(difficulty: str, number: int) -> np.ndarray:
    """
        Read the file and return the sudoku puzzle as a numpy array.
        :param difficulty: The difficulty of the file.
        :param number: The number of the file.
        :return: the sudoku board
    """

    p = f"./datasets/{difficulty}/{difficulty}{number}.txt"
    try:
        # Open the file and read the content
        with open(p, 'r') as file:
            s = file.read()
            # Split the content by line and then by space
            # The result is a 9x9 numpy array
            sudoku = np.array([[int(i) for i in l] for l in s.split()])
            return sudoku

    except FileNotFoundError:
        # If the file is not found, raise an error with a message
        return None



def print_sudoku(board: np.ndarray) -> None:
    """
        Print the sudoku board.
        :param board: The sudoku board
    """
    # Print the upper line of the sudoku puzzle with a horizontal line between each 3x3 box
    # The character "+" is used to mark the corners of each 3x3 box
    print("+" + "-" * 23 + "+")

    # Print the solved sudoku puzzle row by row
    for i in range(9):
        # Print a vertical line at the start of each row
        print("|", end=" ")

        # Print each value in the row
        for j in range(9):
            # Print the value
            print(board[i][j], end=" ")
            # Print a vertical line between each 3x3 box, excluding the last column
            if j % 3 == 2 and j < 8:
                print("|", end=" ")

        # Print a vertical line at the end of each row
        print("|")

        # Print a horizontal line between each 3x3 box and at the end of the sudoku puzzle
        if i % 3 == 2:
            print("+" + "-" * 23 + "+")



def check_no_repetition(row: int, column: int, board: np.ndarray) -> int:
    """
        This function follows the contraints the sudoku game requires, that is,
        1. no two equal digits appear for each row, column, and box;
        2. each digit must appear in each row, column, and box
        :param row: the first coordinate to identify the content of a cell that requires to be checked
        :param column: the second coordinate to identify the content of a cell that requires to be checked
        :param board: the sudoku board
        :return: the numbers of errors in each row, column and box, that is, the number of repetitions of the same value
        in each row, column, box
    """

    # Retrieve all values of a given row
    row_values = board[row, :]

    # Calculate the row_errors: I want the len of the array of the row-th row except for 0-value that describes the absence of a value.
    # Then, I want the len of the non-repeated values except for 0-value. If their difference is zero, no repetitions are present.
    row_errors = len(row_values[row_values != 0]) - len(np.unique(row_values[(row_values != 0)]))

    column_values = board[:, column]
    column_errors = len(column_values[column_values != 0]) - len(np.unique(column_values[column_values != 0]))

    start_row = (row // 3) * 3
    start_col = (column // 3) * 3
    box = board[start_row:start_row + 3, start_col:start_col + 3]

    box_errors = len(box[box != 0]) - len(np.unique(box[(box != 0)]))
    # Sommiamo tutti gli errori
    number_of_errors = row_errors + column_errors + box_errors

    return number_of_errors


def possible_values(board) -> np.ndarray:
    """
        This function fill array with all possible values that can be assigned to a certain sudoku solver
        :param board: the sudoku board
        :return: an array of possible elements
    """

    #This is the values array
    fill_array = []
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:

                # Initialize the values array
                fill_array.append([])
                for s in range(1, 10):
                    if s not in board[i] and s not in [board[x][j] for x in range(9)] and s not in [board[x][y] for x in
                                                                                                    range((i // 3) * 3,
                                                                                                          (i // 3) * 3 + 3)
                                                                                                    for y in
                                                                                                    range((j // 3) * 3,
                                                                                                       (j // 3) * 3 + 3)]:
                        # Append k-th element in the last position, as a queue
                        fill_array[-1].append(s)
    return fill_array


def find_free_cell(board, n) -> tuple[int, int]:
    """
        This function finds the n-th free cell in the board
        :param board: the sudoku board
        :param n: cell to find
        :return: coordinate of the cell (row, col)
    """

    count = 0
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                if count == n:
                    return i, j
                count += 1


#
def solve(current_state) -> bool:
    """
        Function that solves the sudoku board using backtracking and recursion
        :param current_state: the sudoku board
        :return: a boolean value that indicates if the sudoku problem is solved
    """

    # Find all the possibilities for each free cell
    possible_v = possible_values(current_state)

    # If there are no free cells, the board is solved
    if not possible_v:
        return True

    # Find the subarray with the fewest elements
    arr_few_elements = min(possible_v, key=len)
    min_possibilities_index = possible_v.index(arr_few_elements)
    row, col = find_free_cell(current_state, min_possibilities_index)

    # Try each possibility for the array with the fewest elements
    for tryv in arr_few_elements:

        #Try a possible solution
        current_state[row][col] = tryv

        #Use recursion to check if the assigned element is part of final solution; otherwise assign 0-value again and try again with another number
        if solve(current_state):
            return True
        current_state[row][col] = 0

    return False


def complete(current_state) -> bool:
    """
        This function will check if the current state of the sudoku problem is complete
        :param current_state: the sudoku board
        :return: a boolean value that indicates if the sudoku problem is completed
    """

    for row in current_state:
        if 0 in row:
            return False
    return True

def check_if_solved(current_state) -> bool:
    """
        This function check if sudoku problem is solved
        :param current_state: the sudoku board
        :return: a boolean value that indicates if the sudoku problem is solved
    """
    for i in range(9):
        for j in range(9):
            if check_no_repetition(i, j, sudoku_problem) == 0 and complete(sudoku_problem):
                return True
    return False


def calculate_cost(grid) -> int:
    """
        This optimization function calculates the cost of a possible solution
        :param grid: the sudoku board
        :return: the cost number of the solutions: no cost means optimal solution
    """

    cost = 0
    for row in grid:
        cost += 9 - len(set(row))  # Penalty of duplicated values in a row
    for col in zip(*grid):
        cost += 9 - len(set(col))  # Penalty of duplicated values in a column
    for i in range(3):
        for j in range(3):
            block = [grid[x][y] for x in range(i*3, (i+1)*3) for y in range(j*3, (j+1)*3)]
            cost += 9 - len(set(block))  # Penalty of duplicated values in a 3x3 box
    return cost

def make_move(grid, fixed_positions) -> np.ndarray:
    """
        The function tries to change the numbers in some grid positions to explore a new possible solution.
        This exchange represents the “exploration attempt” of the Simulated Annealing algorithm:
        we slightly modify the current grid configuration to see if this small change can lead to a better solution.
        :param grid: the sudoku board
        :param fixed_positions: boolean matrix that represents no-modificable values
        :return: the Sudoku grid after making the move
    """

    new_grid = copy.deepcopy(grid)
    row = random.randint(0, 8)
    cols = [c for c in range(9) if not fixed_positions[row][c]]  # Trova posizioni modificabili
    if len(cols) >= 2:
        col1, col2 = random.sample(cols, 2)
        new_grid[row][col1], new_grid[row][col2] = new_grid[row][col2], new_grid[row][col1]
    return new_grid

def simulated_annealing(grid, fixed_positions, initial_temp=1000, cooling_rate=0.99, max_iterations=1000000) -> np.ndarray:
    """
        This is a probabilistic optimization function that tries to find a solution from an initial configuration,
        making small changes and accepting worse solutions with some probability to avoid getting stuck in local minima.
        For more information, see https://en.wikipedia.org/wiki/Simulated_annealing
        :param grid: the sudoku board
        :param fixed_positions: boolean matrix that represents no-modificable values
        :param initial_temp: the temperature of the algorithm, that is, an initial value that is gradually decreased
        each time the algorithm approaches the optimal solution. If the value takes the value 0, the optimal solution
        has been found
        :param cooling_rate: decreasing factor
        :param max_iterations: max number of attemps to find optimal solution
        :return: sudoku board solved
    """

    c_s = copy.deepcopy(grid)
    current_cost = calculate_cost(c_s)
    temperature = initial_temp

    for i in range(max_iterations):
        if current_cost == 0:
            print(f"Soluzione trovata in {i} iterazioni!")
            return c_s

        # Generate new solution
        new_solution = make_move(c_s, fixed_positions)
        new_cost = calculate_cost(new_solution)

        # Calculate the probability of accepting the new solution
        if new_cost < current_cost or random.uniform(0, 1) < math.exp((current_cost - new_cost) / temperature):
            c_s, current_cost = new_solution, new_cost

        # Cooling
        temperature *= cooling_rate

        # If temperature is too low, stop
        if temperature < 1e-5:
            break

    print("Soluzione non trovata.")
    return c_s


def solve_and_time(board) -> float:
    """
        Solve a sudoku puzzle using Simulated Annealing and return the time it took to solve it.
        :param board: The sudoku board
        :return: The time it took to solve the sudoku puzzle
    """

    # Start the timer
    s = time.time()

    # Solve the sudoku puzzle using constraint satisfaction approach and backtracking
    solve(board)

    # Stop the timer
    e = time.time()

    # Return the time it took to solve the sudoku puzzle in seconds
    return e - s

def count_clues_and_empty(board) -> tuple[int, int]:
    """
        Count the number of clues and empty positions in a sudoku puzzle.
        :param board: The sudoku board
        :return: The number of clues and empty positions in the sudoku puzzle
    """

    # Initialize the number of clues and empty positions to 0
    clues = 0
    empty = 0

    # For each position in the sudoku puzzle, check if it is empty or not
    for i in range(9):
        for j in range(9):
            # If the position is empty, increase the number of empty positions by 1
            if board[i][j] == 0:
                empty += 1
            # Otherwise, increase the number of clues by 1
            else:
                clues += 1

    # Return the number of clues and empty positions in the sudoku puzzle
    return clues, empty

def solve_all(difficulty: str, number: int) -> float:
    """
        Solve all the sudoku puzzles in the given difficulty.
        :param difficulty: The difficulty of the sudoku puzzles.
        :param number: The number of the sudoku puzzles.
    """
    # For each sudoku puzzle in the given difficulty, solve it and print the time it took to solve it
    for i in range(1, number + 1):
        # Open the file containing the sudoku puzzle
        b = open_board(difficulty, str(i))
        if b is None:
            # If the file is not found, skip it
            continue

        # Count the number of clues and empty positions in the sudoku puzzle
        c, e = count_clues_and_empty(b)

        # Solve the sudoku puzzle using constraint satisfaction approach and backtracking
        time = solve_and_time(b)

        # Append the results to the dataframe
        df.loc[len(df)] = [f"{difficulty}{i}", difficulty, c, e, time, "Yes"]


        # Print the time it took to solve the sudoku puzzle
    print(df)
    return time

if __name__ == '__main__':

    time_d = {}

    for i in ["easy", "hard", "medium", "normal"]:
        for j in range(1, 7):

            sudoku_problem = open_board(i, j)
            if(sudoku_problem is not None):
                print_sudoku(sudoku_problem)
                solve(sudoku_problem)

                #Generate boolean of fixed positions
                fixed_positions = [[sudoku_problem[r][c] != 0 for c in range(9)] for r in range(9)]

                if check_if_solved(sudoku_problem):
                    print("Sudoku board solved successfully!")

                    #Call Simulated Annealing
                    solution = simulated_annealing(sudoku_problem, fixed_positions)
                    print_sudoku(sudoku_problem)

                    #Verify execution time for all sudoku problem

                else:
                    print("Sudoku board could not be solved.")
            else:
                print("Sudoku board not found")
        time_c = solve_all(i, len(os.listdir(f"./datasets/{i}")))
        time_d[i] = time_c
        print(f"The average time it took to solve the {i}{j} puzzles is:", time_c, "seconds")


