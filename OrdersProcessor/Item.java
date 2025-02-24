package processor;

//Class that represents an Item within an order
public class Item implements Comparable<Item>{

	private String itemName;
	private double itemCost;
	private int numberOfItems;
	private int numberOfItemsTotal;
	
	public Item(String itemName, double itemCost, int numItems) {
		this.itemName = itemName;
		this.itemCost = itemCost;
		this.numberOfItems = numItems;
		if(numItems == 1) {
			numberOfItemsTotal = numItems;
		}
	}

	public String getItemName() {
		return itemName;
	}


	public double getItemCost() {
		return itemCost;
	}


	public double getCost() {
		return itemCost * numberOfItems;
	}
	
	public double getTotalCost() {
		return itemCost * numberOfItemsTotal;
	}
	
	public void addNumberOfItems() {
		numberOfItemsTotal++;
		numberOfItems++;
	}
	
	public int getNumberOfItems() {
		return numberOfItems;
	}
	
	public int getNumberOfItemsTotal() {
		return numberOfItemsTotal;
	}

	public void setNumberOfItems(int numberOfItems) {
		this.numberOfItems = numberOfItems;
	}
	
	public int compareTo(Item other) {
		return (this.itemName.compareTo(other.itemName));
	}
	
	
}
