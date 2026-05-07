using UnityEngine;
using TMPro;
using System.Collections.Generic;
using System;
using System.Globalization;
using System.IO;

public class WarehouseStatsManager : MonoBehaviour
{
    [Header("UI Reference")]
    public TextMeshProUGUI statsTextDisplay;

    [Header("CSV Logging")]
    public bool enableCsvLogging = true;
    public string csvFileName = "warehouse_stats.csv";
    public int snapshotIntervalSteps = 100;

    [Header("Global Metrics")]
    public int totalElapsedSteps = 0;
    public int teamDeliveries = 0;
    public int totalCollisions = 0;
    public int emptyDropViolations = 0;

    // A class to hold individual agent data
    private class AgentStats
    {
        public string id;
        public int deliveries;
        public int rackHits;
        public int totalStepsTaken;
        public int bestSteps = int.MaxValue;

        public float AvgSteps => deliveries == 0 ? 0 : (float)totalStepsTaken / deliveries;
    }

    private Dictionary<GridAgent, AgentStats> agentData = new Dictionary<GridAgent, AgentStats>();
    private string csvFilePath;

    private void Start()
    {
        InitializeCsvLog();
        UpdateUI();
    }

    // Call this from GridAgent.Awake()
    public void RegisterAgent(GridAgent agent, string id)
    {
        if (!agentData.ContainsKey(agent))
        {
            agentData.Add(agent, new AgentStats { id = id });
        }
    }

    // --- Data Recording Methods ---
    
    public void RecordDelivery(GridAgent agent, int stepsSincePickup)
    {
        teamDeliveries++;
        agentData[agent].deliveries++;
        agentData[agent].totalStepsTaken += stepsSincePickup;
        
        if (stepsSincePickup < agentData[agent].bestSteps)
        {
            agentData[agent].bestSteps = stepsSincePickup;
        }
        LogEvent("delivery", agent, stepsSincePickup);
        UpdateUI();
    }

    public void RecordCollision()
    {
        totalCollisions++;
        LogEvent("collision", null, -1);
        UpdateUI();
    }

    public void RecordRackHit(GridAgent agent)
    {
        agentData[agent].rackHits++;
        LogEvent("rack_hit", agent, -1);
        UpdateUI();
    }

    public void RecordEmptyDrop()
    {
        emptyDropViolations++;
        LogEvent("empty_drop", null, -1);
        UpdateUI();
    }

    // Counts one agent decision. With two agents, two decisions usually equal one team environment tick.
    public void IncrementGlobalStep()
    {
        totalElapsedSteps++;
        
        // Update UI every 10 steps so we don't lag the game loop with string building
        if (totalElapsedSteps % 10 == 0) UpdateUI(); 
        if (snapshotIntervalSteps > 0 && totalElapsedSteps % snapshotIntervalSteps == 0)
        {
            LogEvent("step_snapshot", null, -1);
        }
    }

    private void InitializeCsvLog()
    {
        if (!enableCsvLogging) return;

        csvFilePath = Path.Combine(Application.persistentDataPath, csvFileName);
        string directory = Path.GetDirectoryName(csvFilePath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        if (!File.Exists(csvFilePath))
        {
            File.AppendAllText(csvFilePath,
                "timestamp,global_step,event,agent_id,team_deliveries,agent_deliveries,steps_since_pickup,avg_steps,best_steps,rack_hits,total_rack_hits,collisions,empty_drop_violations,delivery_rate,efficiency\n");
        }
    }

    private void LogEvent(string eventName, GridAgent agent, int stepsSincePickup)
    {
        if (!enableCsvLogging) return;
        if (string.IsNullOrEmpty(csvFilePath)) InitializeCsvLog();

        AgentStats stats = null;
        if (agent != null && agentData.ContainsKey(agent))
        {
            stats = agentData[agent];
        }

        int totalRackHits = GetTotalRackHits();
        float deliveryRate = GetDeliveryRate();
        float efficiency = GetEfficiency(totalRackHits);

        string row = string.Join(",",
            Csv(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture)),
            totalElapsedSteps.ToString(CultureInfo.InvariantCulture),
            Csv(eventName),
            Csv(stats != null ? stats.id : "TEAM"),
            teamDeliveries.ToString(CultureInfo.InvariantCulture),
            (stats != null ? stats.deliveries : 0).ToString(CultureInfo.InvariantCulture),
            stepsSincePickup.ToString(CultureInfo.InvariantCulture),
            FormatFloat(stats != null ? stats.AvgSteps : 0f),
            (stats != null && stats.bestSteps != int.MaxValue ? stats.bestSteps : 0).ToString(CultureInfo.InvariantCulture),
            (stats != null ? stats.rackHits : 0).ToString(CultureInfo.InvariantCulture),
            totalRackHits.ToString(CultureInfo.InvariantCulture),
            totalCollisions.ToString(CultureInfo.InvariantCulture),
            emptyDropViolations.ToString(CultureInfo.InvariantCulture),
            FormatFloat(deliveryRate),
            FormatFloat(efficiency)
        );

        File.AppendAllText(csvFilePath, row + "\n");
    }

    private int GetTotalRackHits()
    {
        int totalRackHits = 0;
        foreach (var kvp in agentData)
        {
            totalRackHits += kvp.Value.rackHits;
        }

        return totalRackHits;
    }

    private float GetDeliveryRate()
    {
        return totalElapsedSteps == 0 ? 0f : (float)teamDeliveries / totalElapsedSteps;
    }

    private float GetEfficiency(int totalRackHits)
    {
        int penaltyWeight = (totalCollisions * 50) + (totalRackHits * 10) + (emptyDropViolations * 5);
        if (totalElapsedSteps + penaltyWeight <= 0)
        {
            return 0f;
        }

        return Mathf.Clamp(((float)teamDeliveries * 1000f) / (totalElapsedSteps + penaltyWeight), 0f, 100f);
    }

    private string FormatFloat(float value)
    {
        return value.ToString("F3", CultureInfo.InvariantCulture);
    }

    private string Csv(string value)
    {
        if (value == null) return "";
        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    // --- UI Formatting ---
    private void UpdateUI()
    {
        if (statsTextDisplay == null) return;

        string uiText = "";
        int totalRackHits = GetTotalRackHits();

        // 1. Per-Agent Breakdown
        foreach (var kvp in agentData)
        {
            AgentStats s = kvp.Value;
            uiText += $"{s.id} Deliveries: {s.deliveries} | Avg Steps: {s.AvgSteps:F1} | Rack Hits: {s.rackHits}\n";
        }

        uiText += "\n"; // Spacer

        // 2. Team Metrics
        float deliveryRate = GetDeliveryRate();
        float efficiency = GetEfficiency(totalRackHits);

        uiText += $"Team Deliveries: {teamDeliveries}\n";
        uiText += $"Collisions: {totalCollisions}\n";
        uiText += $"Delivery Rate: {deliveryRate:F3} / step\n";
        uiText += $"Efficiency: {efficiency:F1}%\n";

        statsTextDisplay.text = uiText;
    }
}
