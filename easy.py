








# 746. Min Cost Climbing Stairs
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        cost.append(0)
        ln = len(cost)
        f = [0] * ln
        f[0] = cost[0]
        f[1] = cost[1]
        for i in range(2,ln):
            f[i] = min(f[i-1],f[i-2]) + cost[i]
        return f[-1]


# 70. Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        f = [1] * n
        if n < 2:
            return 1
        f[1] = 2
        for i in range(2,n):
            f[i] = f[i-1] + f[i-2]
        return f[-1]


# 121. Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        
        p = prices[1] - prices[0]
        b = prices[0]
        for i in range(1,len(prices)):
            p = max(prices[i] - b, p)
            b = min(prices[i],b)
        return max(p,0)


# 1025. Divisor Game
class Solution:
    def divisorGame(self, N: int) -> bool:
        f = [False] * (N+1)
        for i in range(2, N+1):
            for x in range(i,0,-1):
                if i % x == 0:
                    f[i] = not f[i-x]
        return f[N]


# 53. Maximum Subarray
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        lm = nums[0]
        gm = nums[0]
        for i in range(1,len(nums)):
            lm = max(nums[i],lm + nums[i])
            gm = max(lm,gm)
        return gm


# 53. Maximum Subarray  # this is a algrith
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ln = len(nums)
        f = [nums[0]] * ln
        for i in range(1,ln):
            f[i] = max(nums[i],f[i-1]+nums[i])
            print(f[i])
        return max(f)


# 38. Count and Say
class Solution:
    def countAndSay(self, n: int) -> str:
        def recu(s:str):
            r = ""
            ct = 0
            t = s[0]
            for i in s:
                if i == t:
                    ct += 1
                else:
                    r += str(ct) + str(t)
                    t = i
                    ct = 1
            r += str(ct) + str(t)
            return r
    
        t = '1'
        for i in range(n-1):
            t = recu(t)
        return t


# 35. Search Insert Position
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        for i in range(len(nums)):
            if target <= nums[i]:
                return i
        return len(nums)


# 28. Implement strStr()
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle == "" :
            return 0
        if len(needle) > len(haystack):
            return -1
        
        ln = len(needle)
        lh = len(haystack)
        for i in range(lh - ln + 1):
            if haystack[i:i+ln] == needle:
                return i
        return -1


# 27. Remove Element
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        while(val in nums):
            del nums[nums.index(val)]  # or nums.remove(val)
        return len(nums)


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
        