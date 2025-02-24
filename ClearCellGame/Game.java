package model;

/**
 * This class represents the logic of a game where a board is updated on each
 * step of the game animation. The board can also be updated by selecting a
 * board cell.
 */

public abstract class Game {
	protected BoardCell[][] board;

	/**
	 * Defines a board with BoardCell.EMPTY cells.
	 */
	public Game(int maxRows, int maxCols) {
		board = new BoardCell[maxRows][maxCols];
		for (int row = 0; row < board.length; row++) {
			for (int col = 0; col < board[row].length; col++) {
				board[row][col] = BoardCell.EMPTY;
			}
		}
		
	}

	public int getMaxRows() {
		return board.length;
	}

	public int getMaxCols() {
		return board[0].length;
	}

	public void setBoardCell(int rowIndex, int colIndex, BoardCell boardCell) {
		board[rowIndex][colIndex] = boardCell;
	}

	public BoardCell getBoardCell(int rowIndex, int colIndex) {
		return board[rowIndex][colIndex];
	}

	/**
	 * Initializes row with the specified color.
	 */
	public void setRowWithColor(int rowIndex, BoardCell cell) {
		for (int i = 0; i < board[rowIndex].length; i++) {
			board[rowIndex][i] = cell;
		}
	}
	
	/**
	 * Initializes column with the specified color.
	 */
	public void setColWithColor(int colIndex, BoardCell cell) {
		for (int i = 0; i < board.length; i++) {
			board[i][colIndex] = cell;
		}
	}
	
	/**
	 * Initializes the board with the specified color.
	 */
	public void setBoardWithColor(BoardCell cell) {
		for (int row = 0; row < board.length; row++) {
			for (int col = 0; col < board[row].length; col++) {
				board[row][col] = cell;
			}
		}
	}	
	
	public abstract boolean isGameOver();

	public abstract int getScore();

	/**
	 * Advances the animation one step.
	 */
	public abstract void nextAnimationStep();

	/**
	 * Adjust the board state according to the current board state and the
	 * selected cell.
	 */
	public abstract void processCell(int rowIndex, int colIndex);
}