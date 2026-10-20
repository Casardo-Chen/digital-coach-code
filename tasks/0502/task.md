## Task Name
Generate insights using VLOOKUP

#### Task Description
Your manager requires a "Net Revenue" report that accounts for varying tax rates across product categories. Currently, your Sales Log is missing prices, and your Product Catalog is missing tax rates. Furthermore, tax rates are stored in a third Tax Table based on category (e.g., Electronics = 10%, Furniture = 5%).

You will use the VLOOKUP function to create a live link between the Sales Log and the Master Inventory. By the end of this task, entering a Product ID will automatically "fetch" the correct price and category, allowing you to calculate the total revenue for all 100 rows in seconds. You will then "nest" that result inside another calculation to pull the correct tax rate, and then create a column that includes the net revenue per transaction.

#### Learning Objectives
- Identify the lookup value that links two tables.
- Use VLOOKUP with a fixed table array.
- Choose the correct column index number.
- Use FALSE or 0 for exact matching.
- Troubleshoot common lookup errors such as #N/A.

#### Final Outcome
A completed Sales Log where each row automatically includes product price, category, tax rate, tax amount, and net revenue using VLOOKUP-based formulas.