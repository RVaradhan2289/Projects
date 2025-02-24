package processor;

import java.io.*;
import java.text.NumberFormat;
import java.util.*;
import java.util.Scanner;

//Class that processes orders of items into a sorted purchase order

public class OrdersProcessor {

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		System.out.println("Enter item's data file name: ");
		String fileName = scanner.nextLine();
		File itemFile = new File(fileName);
		System.out.println("Enter 'y' for multiple threads, any other character"
				+ " otherwise: ");
		String threadType = scanner.nextLine();
		System.out.println("Enter number of orders to process: ");
		int times = scanner.nextInt();
		scanner.nextLine();
		System.out.println("Enter order's base filename: ");
		String baseName = scanner.nextLine();
		System.out.println("Enter result's filename: ");
		String resultName = scanner.nextLine();
		File resultFile = new File(resultName);
		
		//starts data processing runtime
		long startTime = System.currentTimeMillis();
		if(threadType.equals("y")) {
			Order.itemList = new TreeMap<>();
			Order[] orders = new Order[times];
			Thread[] orderThreads = new Thread[times];
			/*Reads the list of possible items in an order and saves it to 
			 *itemList
			 */
			try {
				Scanner itemFileReader = new Scanner(new FileReader(itemFile));
				while(itemFileReader.hasNextLine()) {
					String currentLine = itemFileReader.nextLine();
					String itemName = currentLine.split(" ")[0].trim();
					double itemPrice = Double.parseDouble(currentLine.split(" ")
							[1].trim());
					Order.itemList.put(itemName, new Item(itemName, itemPrice, 
							0));
				}
				//For every order, reads order file and adds items to orderList  
				for(int i = 0; i < times; i++) {
					File baseFile = new File(baseName + (i + 1) + ".txt");
					int orderNumber = 0;
					try {
						Scanner baseFileReader = new Scanner(new 
								FileReader(baseFile));
						if(baseFileReader.hasNextLine()) {
							String currentLine = baseFileReader.nextLine();
							orderNumber = Integer.parseInt(currentLine.
									split(":")[1].trim());
							orders[i] = new Order(orderNumber);
						}
						while(baseFileReader.hasNextLine()) {
							String currentLine = baseFileReader.nextLine();
							String baseItemName = currentLine.
									split(" ")[0].trim();
							if(orders[i].orderList.containsKey(baseItemName)) {
								Order.itemList.get(baseItemName)
								.addNumberOfItems();
								orders[i].orderList.get(baseItemName).
								addNumberOfItems();
							} else {
								double itemPrice = Order.itemList.get
										(baseItemName).getItemCost();
								orders[i].add(new Item(baseItemName, itemPrice, 
										1));
							}
						}
					} catch (FileNotFoundException e) {
						e.printStackTrace();
					}
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			//Creates threads for each order
			for(int i = 0; i < times; i++) {
				orderThreads[i] = new Thread(orders[i]);
				orderThreads[i].start();
			}
			//Joins threads
			try {
				for(int i = 0; i < times; i++) {
					orderThreads[i].join();
				} 
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			//Writes summary after all orders are processed
			try {
				FileWriter addToFile = new FileWriter(resultFile);
				double summaryCost = 0.0;
				for(String order: Order.multiThreadList.values()) {
					addToFile.append(order);
				}
				addToFile.append("***** Summary of all orders *****\n");
				for(Item item : Order.itemList.values()) {
					int summaryCount = 0;
					for(int i = 0; i < times; i++) {
						Item currItem = item;
						if(orders[i].orderList.get
								(item.getItemName()) == null) {
							continue;
						} else {
							currItem = orders[i].orderList.get
									(item.getItemName());
						}
						summaryCount += currItem.getNumberOfItems();
					}
					String getItemCost = NumberFormat.getCurrencyInstance()
							.format(item.getItemCost());
					String getTotalCost = NumberFormat.getCurrencyInstance().
							 format(item.getItemCost() * summaryCount);
					addToFile.append("Summary - Item's name: " + 
				item.getItemName() + 
							", Cost per item: " + getItemCost + 
							", Number sold: " + summaryCount + ", "
									+ "Item's Total: " + getTotalCost + "\n");
					summaryCost += (item.getItemCost() * summaryCount);
				}
				String summaryCostReformat = NumberFormat.getCurrencyInstance()
						.format(summaryCost);
				addToFile.append("Summary Grand Total: " + 
							 summaryCostReformat + "\n");
				addToFile.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		} else {
			processOrder(resultFile, baseName, itemFile, times);
		}
		long endTime = System.currentTimeMillis();
		System.out.println("Processing time (msec): " + (endTime - startTime));
		System.out.println("Results can be found in the file: " + resultFile);
		scanner.close();
	}

	//Method that processes orders under in a single thread
	public static void processOrder(File addToFile, String baseName, 
			File itemFile, int index) {
		int orderNumber = 0;
		Order.itemList = new TreeMap<>();
		//Reads item list and adds possible items to itemList
		try {
			Scanner itemFileReader = new Scanner(new FileReader(itemFile));
			while(itemFileReader.hasNextLine()) {
				String currentLine = itemFileReader.nextLine();
				String itemName = currentLine.split(" ")[0].trim();
				double itemPrice = Double.parseDouble(currentLine.split(" ")
						[1].trim());
				Order.itemList.put(itemName, new Item(itemName, itemPrice, 0));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		try {
			FileWriter addToFileWriter = new FileWriter(addToFile);
			/* For every order, checks the number of occurences of an item in 
			 * the order and updates itemList with correct total
			 */
			for (int i = 1; i <= index; i++) {
				double totalCost = 0.0;
				for(Item item : Order.itemList.values()) {
					item.setNumberOfItems(0);
				}
				File baseFile = new File(baseName + i + ".txt");
				try {
					Scanner baseFileReader = new Scanner(new 
							FileReader(baseFile));
					if(baseFileReader.hasNextLine()) {
						String currentLine = baseFileReader.nextLine();
						orderNumber = Integer.parseInt(currentLine.split(":")
								[1].trim());
					}
					while(baseFileReader.hasNextLine()) {
						String currentLine = baseFileReader.nextLine();
						String baseItemName = currentLine.split(" ")[0].trim();
						for(Item item : Order.itemList.values()) {
							if(item.getItemName().equals(baseItemName)) {
								item.addNumberOfItems();
							}
						}
					}
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
				//Writes order using itemList details 
				addToFileWriter.append("----- Order details for client with "
						+ "Id: " + orderNumber + " -----\n");
				for(Item item : Order.itemList.values()) {
					String getItemCost = NumberFormat.getCurrencyInstance()
							.format(item.getItemCost());
					String getCost = NumberFormat.getCurrencyInstance().
							 format(item.getCost());
					if(item.getNumberOfItems() == 0) {
						continue;
					}
					addToFileWriter.append("Item's name: " + item.getItemName() 
					+ ", Cost per item: " + getItemCost + ", Quantity: " 
							+ item.getNumberOfItems() + ", Cost: " + getCost 
							+ "\n");
					totalCost += item.getCost();
				}
				String totalCostReformat = NumberFormat.getCurrencyInstance().
						format(totalCost);
				addToFileWriter.append("Order Total: " + totalCostReformat
						 + "\n");
			}
			double summaryCost = 0.0;
			//Writes summary of all orders totaled
			addToFileWriter.append("***** Summary of all orders *****\n");
			for(Item item : Order.itemList.values()) {
				String getItemCost = NumberFormat.getCurrencyInstance()
						.format(item.getItemCost());
				String getTotalCost = NumberFormat.getCurrencyInstance().
						 format(item.getTotalCost());
				addToFileWriter.append("Summary - Item's name: " + 
			item.getItemName() + 
						", Cost per item: " + getItemCost + 
						", Number sold: " + item.getNumberOfItemsTotal() + ", "
								+ "Item's Total: " + getTotalCost + "\n");
				summaryCost += item.getTotalCost();
			}
			String summaryCostReformat = NumberFormat.getCurrencyInstance()
					.format(summaryCost);
			addToFileWriter.append("Summary Grand Total: " + 
						 summaryCostReformat + "\n");
			addToFileWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
