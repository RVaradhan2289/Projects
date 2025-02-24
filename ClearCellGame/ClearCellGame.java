package model;

import java.util.Random;

/* Game logic for Clear Cell Game */
public class ClearCellGame extends Game {

	private Random random;
	private static int points;
	private static int methodCallNumber;

	public ClearCellGame(int maxRows, int maxCols, Random random, 
			int strategy) {
		super(maxRows, maxCols);
		this.random = random;
		points = 0;
		methodCallNumber = 0;
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
		if (board[rowIndex][colIndex] == BoardCell.EMPTY) {
			return;
		}
		BoardCell cellColor = board[rowIndex][colIndex];
		methodCallNumber++;
		board[rowIndex][colIndex] = BoardCell.EMPTY;
		if (rowIndex != 0) {
			processCellHelper(cellColor, rowIndex, colIndex, -1, 0);
		}
		if (rowIndex != board.length - 1) {
			processCellHelper(cellColor, rowIndex, colIndex, 1, 0);
		}
		if (colIndex != 0) {
			processCellHelper(cellColor, rowIndex, colIndex, 0, -1);
			points++;
		}
		if (colIndex != board[0].length - 1) {
			processCellHelper(cellColor, rowIndex, colIndex, 0, 1);
		}
		if (colIndex != 0 && rowIndex != 0) {
			processCellHelper(cellColor, rowIndex, colIndex, -1, -1);
		}
		if (colIndex != 0 && rowIndex != board.length - 1) {
			processCellHelper(cellColor, rowIndex, colIndex, 1, -1);
		}
		if (colIndex != board[0].length - 1 && rowIndex != 0) {
			processCellHelper(cellColor, rowIndex, colIndex, -1, 1);
		}
		if (colIndex != board[0].length - 1 && rowIndex != board.length - 1) {
			processCellHelper(cellColor, rowIndex, colIndex, 1, 1);
		}
		methodCallNumber--;
		points++;
		if (methodCallNumber == 0) {
			removeRow();
		}
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

	private void processCellHelper(BoardCell cellColor, int rowIndex, 
			int colIndex, int rowCounter, int colCounter) {
		while (cellColor == board[rowIndex + rowCounter][colIndex + colCounter]){
			processCell(rowIndex + rowCounter, colIndex + colCounter);
			board[rowIndex + rowCounter][colIndex + colCounter] = BoardCell.EMPTY;
			if (rowIndex + rowCounter == 0 || rowIndex + rowCounter == 
					board.length - 1 || colIndex + colCounter == 0 || colIndex 
					+ colCounter == board[0].length) {
				return;
			}
		}
	}
}


