package processor;

import java.text.NumberFormat;
import java.util.*;

//Class that represents an order in the processor
public class Order implements Runnable{

	protected static TreeMap<String,Item> itemList = new TreeMap<>();
	protected static TreeMap<Integer,String> multiThreadList = new TreeMap<>();
	protected TreeMap<String, Item> orderList;
	private int orderNumber;

	public Order(int orderNumber) {
		orderList = new TreeMap<String,Item>();
		this.orderNumber = orderNumber;
	}

	public int getOrderNumber() {
		return orderNumber;
	}

	public void setOrderNumber(int orderNumber) {
		this.orderNumber = orderNumber;
	}

	public void add(Item item) {
		orderList.put(item.getItemName(), item);
	}

	//Processes one order's(thread's) order
	public void run() {
		//locks use of multiThreadList map to one thread at a time
		synchronized(multiThreadList) {
			String order = "";
			double totalCost = 0.0;
			order += ("----- Order details for client with Id:"
					+ " " + orderNumber + " -----\n");
			for(Item item : orderList.values()) {
				String getItemCost = NumberFormat.getCurrencyInstance()
						.format(item.getItemCost());
				String getCost = NumberFormat.getCurrencyInstance().
						format(item.getCost());
				order += ("Item's name: " + item.getItemName() 
				+ ", Cost per item: " + getItemCost + ", Quantity: " 
				+ item.getNumberOfItems() + ", Cost: " + getCost 
				+ "\n");
				totalCost += item.getCost();
			}
			String totalCostReformat = NumberFormat.getCurrencyInstance().
					format(totalCost);
			order += ("Order Total: " + totalCostReformat
					+ "\n");
			multiThreadList.put(orderNumber, order);
		}
	}
}

