













# 26. Remove Duplicates from Sorted Array
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if (nums == []):
            return 0
        
        ct = 1
        t = nums[0]
        for i in nums:
            if(i != t):
                nums[ct] = i
                ct += 1
                t = i
        return ct



# 21. Merge Two Sorted Lists //singly-linked list.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        r = ListNode(-1)
        p = r
        while(l1 and l2):
            if(l1.val < l2.val):
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            p = p.next
        if(l1 == None):
            p.next = l2
        else:
            p.next = l1
            
        return r.next


# 20. Valid Parentheses    ///stack
class Solution:
    def isValid(self, s: str) -> bool:
        if(s==""):
            return True
        
        p = {
            ')':'(',
            '}':'{',
            ']':'['
        }
        stack = []
        for i in s:
            if i in p.values():
                stack.append(i)
            else:
                if(stack==[] or stack.pop() != p[i]):
                    return False
        
        if(stack==[]):
            return True
                    
                        


# 14. Longest Common Prefix
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if (len(strs) == 0):
            return ''
        l = len(strs[0])
        for str in strs:
            if (len(str) < l):
                l = len(str)
        
        counter = 0
        for i in range(l):
            s = strs[0][i]
            for str in strs:
                if(s!=str[i]):
                    return strs[0][0:counter]
            counter+=1
            
        return strs[0][0:counter]
                


# 9. Palindrome Number
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if(x < 0):
            return False
        init = x
        a = 0
        while(x != 0):
            a *= 10
            a += x % 10
            x = x // 10
        print(a)
        if(a == init):
            return True
        else:
            return False


# 13. Roman to Integer
class Solution:
    def romanToInt(self, s: str) -> int:
        dic = {
            'I':1,
            'V':5,
            'X':10,
            'L':50,
            'C':100,
            'D':500,
            'M':1000
        }
        a = 0
        for i in range(0,len(s)-1):
            if(dic[s[i]] < dic[s[i+1]]):
                a -= dic[s[i]]
            else:
                a += dic[s[i]]
            
        return a + dic[s[len(s)-1]]
                

# 7. Reverse Integer
class Solution:
    def reverse(self, x: int) -> int:
        
        sign = 1
        
        if(x < 0):
            x = -x
            sign = -1
            
        a = 0
        while(x != 0):
            a *= 10
            a += x % 10
            x = x // 10
        x = a * sign
        
        if(x < -2**31 or x > 2**31 - 1):
            return 0 
        else:
            return x
        

#1. Two Sum
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        ret = []
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                if(nums[i] + nums[j] == target):
                    ret.append(i)
                    ret.append(j)
        return ret
        