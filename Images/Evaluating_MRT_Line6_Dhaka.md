# Evaluating MRT Line-6 Scheduling in Dhaka: A Dynamic, Data-Driven Approach through Real-Time Passenger Volume: Estimation using Genetic Algorithm and Deep Learning Methods

## Our Developed Tool -

[https://github.com/ahmfuad/metro-scheduler/](https://github.com/ahmfuad/metro-scheduler/)

![photo_2025-07-24_19-41-04.jpg](Evaluating%20MRT%20Line-6%20Scheduling%20in%20Dhaka%20A%20Dynami%20233499b277fa800f9b51c9d1b6c93b42/photo_2025-07-24_19-41-04.jpg)

![photo_2025-07-24_19-41-27.jpg](Evaluating%20MRT%20Line-6%20Scheduling%20in%20Dhaka%20A%20Dynami%20233499b277fa800f9b51c9d1b6c93b42/photo_2025-07-24_19-41-27.jpg)

![photo_2025-07-24_19-42-48.jpg](Evaluating%20MRT%20Line-6%20Scheduling%20in%20Dhaka%20A%20Dynami%20233499b277fa800f9b51c9d1b6c93b42/photo_2025-07-24_19-42-48.jpg)

![photo_2025-07-24_19-41-54.jpg](Evaluating%20MRT%20Line-6%20Scheduling%20in%20Dhaka%20A%20Dynami%20233499b277fa800f9b51c9d1b6c93b42/photo_2025-07-24_19-41-54.jpg)

## Abstract

The introduction of Dhaka Mass Rapid Transit has revolutionized the commuting landscape of Dhaka city, acting as a vital circulatory mode for urban mobility. But unlike the circulatory system, the MRT operates on a fixed pre-programmed schedule that does not adapt to changing demand. This might lead to higher operational cost and lower revenue for operators and higher waiting time for [passengers. So](http://passengers.so/), this begs a very important question, is the current MRT scheduling the most optimal scheduling we have, or can we do better? To find the answer to this question, this study explores a dynamic, data-driven approach to MRT scheduling by comparing the existing static schedule with an optimized model developed through a genetic algorithm. Using MRT pass data and platform surveillance footage, we estimate passenger volumes at each station through computer vision techniques. Then through genetic algorithms we propose an optimal scheduling through dynamic headway for each train. Finally we compare our proposed dynamic schedule with the current static schedule using our simulator by comparing the passenger waiting cost and the operation cost of the MRT for each case.  Our current work includes study on the MRT Line-6 but this can be applied over the whole planned metro routes. This research contributes to the evolving field of intelligent urban transport planning and offers a practical step toward making Dhaka’s transit system more responsive, efficient, and sustainable.

## Methodology

### Passenger Volume Estimation

First we try to count the number of people in a platform to find the current passenger volume of a certain station heading towards a certain direction. We take into account the metro card punches and the surveillance camera feed.
We utilize Deep Learning models to estimate headcount in crowded environments, such as station platforms. By combining Rapid Pass data with surveillance camera feeds, we determine the number of passengers present at a specific
station and heading in a particular direction.

We use RetinaNet to count the number of heads. But RetinaNet fails to give us accurate data if the environment becomes too crowded. If it detects more than 50 heads, we move to CSRNet to get a more accurate result.

![image.png](Evaluating%20MRT%20Line-6%20Scheduling%20in%20Dhaka%20A%20Dynami%20233499b277fa800f9b51c9d1b6c93b42/image.png)

### Optimization

- VoT Estimation:
    
    $$
    VoT = \frac{\text{Monthly Income}}{\text{Number of Weeks}}\div(\text{Working Hours} \times 60) \times 1.5
    $$
    
    - Monthly Income = 29700 (Shaik, 2025)
    - Working Hour = 48.8 Hours (Talukder & Tuz Zahra, 2023)
    
    The VoT came out to be **3.5 tk/min**, from which, waiting cost and traveling cost was calculated.
    
- Operational Cost Estimation:
    
    $$
    \text{Operational Cost} = \frac{\text{Total Cost Per Day}}{\text{Active Hours} \times \text{Number of Trips} \times 60}
    $$
    
    - Daily Total Operational Cost = 2.33 Crore (Talukder & Tuz Zahra, 2023)
    - Number of Trips = 190 (Observer Online Desk, 2024)
    
    We got the total daily operational cost for a certain period and the number of trips for that time period. From that we calculated the operational cost per minute which came out to be **146 tk/min.**
    

![Screenshot from 2025-07-25 02-33-09.png](Evaluating%20MRT%20Line-6%20Scheduling%20in%20Dhaka%20A%20Dynami%20233499b277fa800f9b51c9d1b6c93b42/Screenshot_from_2025-07-25_02-33-09.png)

(“Scheduling Combination and Headway Optimization of Bus Rapid Transit,” 2008)

## Findings

A Passenger flow dataset was generated using the following parameters.

- We know the total number of passengers each day (Railway Supply, 2025)
- The peak hours were found out using the current Metro Schedule. (*ঢাকা ম্যাস ট্রানজিট কোম্পানি লিমিটেড (ডিএমটিসিএল)*, n.d.)
- The busiest stations were found out (Hossain, 2025) and assigned a business weight.
    - Busiest and Least busy stations: [https://en.prothomalo.com/bangladesh/city/c4664wsnb](https://en.prothomalo.com/bangladesh/city/c4664wsnb6)
    
    The business weights of each station -
    
    | Station | Business Weight |
    | --- | --- |
    | Motijheel | 2 |
    | Secretariat | 1 |
    | Dhaka University | 1.5 |
    | Shahbag | 1 |
    | Karwan Bazar | 1.5 |
    | Farmgate | 1.5 |
    | Bijoy Shwarani | 0.5 |
    | Agargaon | 1,5 |
    | Shewrapara | 1 |
    | Kazipara | 1 |
    | Mirpur 10 | 2 |
    | Mirpur 11 | 1 |
    | Pallabi | 1 |
    | Uttara South | .5 |
    | Uttara Center | .5 |
    | Uttara North | 2 |
    
    The dataset -
    
    [metro_weighted_station_passengers](Evaluating%20MRT%20Line-6%20Scheduling%20in%20Dhaka%20A%20Dynami%20233499b277fa800f9b51c9d1b6c93b42/metro_weighted_station_passengers%2023a499b277fa814d9c56fe44b636d9c8.csv)
    

## Findings

![Screenshot from 2025-07-24 20-37-35.png](Evaluating%20MRT%20Line-6%20Scheduling%20in%20Dhaka%20A%20Dynami%20233499b277fa800f9b51c9d1b6c93b42/Screenshot_from_2025-07-24_20-37-35.png)

- Reduced total cost by 17.6%
- Also, significantly reduced wasted working hours and environmental impact

## References

- Shaik, S. (2025, April 2). What Is The Average Salary In Bangladesh: Overview & Insights. *Time Champ - Time and Productivity Tracker*. https://www.timechamp.io/blogs/average-salary-in-bangladesh/
- Talukder, M. S. H., & Tuz Zahra, F. (2023, June 21). *What work‑life balance? Giving employees more control over their work schedules and environments helps reduce the negative effects of long hours and difficult workloads.* Dhaka Tribune. [https://www.dhakatribune.com/opinion/op-ed/313958/what-work-life-balance](https://www.dhakatribune.com/opinion/op-ed/313958/what-work-life-balance)
- Talukder, M. S. H., & Tuz Zahra, F. (2023, June 21). *What work‑life balance? Giving employees more control over their work schedules and environments helps reduce the negative effects of long hours and difficult workloads.* Dhaka Tribune. [https://www.dhakatribune.com/opinion/op-ed/313958/what-work-life-balance](https://www.dhakatribune.com/opinion/op-ed/313958/what-work-life-balance)
- Observer Online Desk. (2024, March 27). *Dhaka Metro Rail operations to continue past 9 pm from Wednesday*. The Daily Observer. [https://www.observerbd.com/news/466118](https://www.observerbd.com/news/466118)
- Scheduling Combination and Headway Optimization of Bus Rapid Transit. (2008). In *JOURNAL OF TRANSPORTATION SYSTEMS ENGINEERING AND INFORMATION TECHNOLOGY* (Vol. 8, Issue 5) [Research paper]. https://doi.org/10.1016/S1570-6672(08)60039-2
- Railway Supply. (2025, February 16). *Dhaka Metro breaks record, transporting 403,164 passengers in a single day.* [https://www.railway.supply/en/dhaka-metro-breaks-record-transporting-403164-passengers-in-a-single-day/](https://www.railway.supply/en/dhaka-metro-breaks-record-transporting-403164-passengers-in-a-single-day/)
- *ঢাকা ম্যাস ট্রানজিট কোম্পানি লিমিটেড (ডিএমটিসিএল)*. (n.d.). https://dmtcl.gov.bd/site/page/d95a6907-4278-4a36-8a90-ee38c2dd43e8/%E0%A6%B8%E0%A6%AE%E0%A7%9F%E0%A6%B8%E0%A7%82%E0%A6%9A%E0%A6%BF
- Hossain, A. (2025, July 9). *The busiest and least used metro stations*. Prothom Alo English. [https://en.prothomalo.com/bangladesh/city/c4664wsnb6](https://en.prothomalo.com/bangladesh/city/c4664wsnb6)