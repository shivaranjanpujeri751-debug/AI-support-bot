orders_database = {
    "ORD001": {
        "customer_name": "John Doe",
        "product": "Laptop",
        "amount": "$999",
        "status": "Delivered",
        "order_date": "2025-12-15",
        "delivery_date": "2025-12-20",
    },
    "ORD002": {
        "customer_name": "Jane Smith",
        "product": "Smartphone",
        "amount": "$599",
        "status": "In Transit",
        "order_date": "2025-12-20",
        "expected_delivery": "2025-12-25",
    },
    "ORD003": {
        "customer_name": "Bob Johnson",
        "product": "Headphones",
        "amount": "$199",
        "status": "Processing",
        "order_date": "2025-12-23",
        "expected_delivery": "2025-12-28",
    },
}

def get_order_details(order_id: str):
    if not order_id:
        return None
    return orders_database.get(order_id.upper())
