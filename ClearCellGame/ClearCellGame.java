package model;

import java.util.Random;

/* Game logic for Clear Cell Game */
public class ClearCellGame extends Game {

	private Random random;
	private static int points;

	public ClearCellGame(int maxRows, int maxCols, Random random, 
			int strategy) {
		super(maxRows, maxCols);
		this.random = random;
		points = 0;
	}
	
	public boolean isGameOver() {
		for (int col = 0; col < board[0].length; col++) {
			if (board[board.length - 1][col] != BoardCell.EMPTY) {
				return true;
			}
		}
		return false;
	}

	public int getScore() {
		return points / 2;
	}

	public void nextAnimationStep() {
		if (!isGameOver()) {
			for (int col = 0; col < board[0].length; col++) {
				if (board[board.length - 1][col] != BoardCell.EMPTY) {
					return;
				}
			}
			BoardCell[][] addRow = new BoardCell[board.length][board[0].length];
			for (int row = 1; row < board.length; row++) {
				for (int col = 0; col < board[row].length; col++) {
					addRow[row][col] = board[row - 1][col];
				}
			}
			for (int col = 0; col < board[0].length; col++) {
				addRow[0][col] = 
						BoardCell.getNonEmptyRandomBoardCell(random); 
			}
			board = addRow;
		}
	}

	public void processCell(int rowIndex, int colIndex) {
		int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
		
		if (board[rowIndex][colIndex] == BoardCell.EMPTY) {
			return;
		}
		
		BoardCell cellColor = board[rowIndex][colIndex];
		board[rowIndex][colIndex] = BoardCell.EMPTY;
		points++;
		
		for (int i = 0; i < 4; i++) {
			int newRow = rowIndex + directions[i][0];
			int newCol = colIndex + directions[i][1];
			
			if (newRow >= 0 && newRow < board.length && newCol >= 0 && newCol < board[0].length) {
				if (board[newRow][newCol] == cellColor) {
					processCell(newRow, newCol);
				}
			}
		}
		
		removeRow();
	}


	private void removeRow() {
		int emptyRow = 0;
		for (int row = 0; row < board.length - 1; row++) {
			for (int col = 0; col < board[row].length; col++) {
				if (board[row][col] == BoardCell.EMPTY) {
					emptyRow++;
				}
				if (emptyRow == board[row].length) {
					BoardCell[][] emptyRowArray = new 
							BoardCell[board.length][board[0].length];
					for (int rowTop = 0; rowTop < row; rowTop++) {
						for (int colTop = 0; colTop < board[rowTop].length; colTop++) {
							emptyRowArray[rowTop][colTop] = board[rowTop][colTop];
						}
					}
					for (int rowBot = row + 1; rowBot < board.length; rowBot++) {
						for (int colBot = 0; colBot < board[rowBot].length; colBot++) {
							emptyRowArray[rowBot - 1][colBot] = board[rowBot][colBot];
						}
					}
					for (int finalCol = 0; finalCol < board[0].length; finalCol++) {
						emptyRowArray[emptyRowArray.length - 1][finalCol] = BoardCell.EMPTY;
					}
					board = emptyRowArray;
				}
			}
			emptyRow = 0;
		}
	}
}


