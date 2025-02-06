class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        carry = 0
        dummy_head = ListNode()
        current = dummy_head

        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0

            # Compute the sum and carry
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10

            # Add the resultant digit to the new linked list
            current.next = ListNode(digit)
            current = current.next

            # Move to the next nodes if they are present
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        return dummy_head.next

# Function to print the linked list
def print_linked_list(node):
    while node:
        print(node.val, end=' ')
        node = node.next
    print()  # New line

# Helper function to create a linked list from a list of numbers
def create_linked_list(numbers):
    dummy_head = ListNode()
    current = dummy_head
    for number in numbers:
        current.next = ListNode(number)
        current = current.next
    return dummy_head.next

# Example with [0, 3, 0] and [0, 4, 0, 0, -1]
list1 = create_linked_list([0, 3, 0])
list2 = create_linked_list([0, 4, 0, 0, -1])
solution = Solution()
result = solution.addTwoNumbers(list1, list2)
print("Result linked list for [0, 3, 0] + [0, 4, 0, 0, -1]:")
print_linked_list(result)

# Another example with [0, 0, 3] and [7]
list1 = create_linked_list([0, 0, 3])
list2 = create_linked_list([7])
result = solution.addTwoNumbers(list1, list2)
print("\nResult linked list for [0, 0, 3] + [7]:")
print_linked_list(result)

# Example with [9, 9, 9] and [1] (checking carry handling)
list1 = create_linked_list([9, 9, 9])
list2 = create_linked_list([1])
result = solution.addTwoNumbers(list1, list2)
print("\nResult linked list for [9, 9, 9] + [1]:")
print_linked_list(result)
