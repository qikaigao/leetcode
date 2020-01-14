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
        