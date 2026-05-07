using UnityEngine;
using System.Collections.Generic;

public class WarehouseGridManager : MonoBehaviour
{
    [Header("Grid Dimensions")]
    public int rows = 10;
    public int cols = 20;
    public float surfaceLevelY = 0f;

    [Header("Fixed Zones")]
    public Vector2Int deliveryLocation = new Vector2Int(1, 1);

    [Header("Agent Tracking")]
    public List<GridAgent> activeAgents = new List<GridAgent>();

    [Header("Cargo Management")]
    public List<Vector2Int> cargoSpawnLocations = new List<Vector2Int>();
    public GameObject cargoPrefab;

    [Header("Collision Rewards")]
    public float globalCollisionPenalty = 25.0f;

    private readonly Dictionary<GridAgent, Vector2Int> cargoLocations = new Dictionary<GridAgent, Vector2Int>();
    private readonly Dictionary<GridAgent, GameObject> activeCargoInstances = new Dictionary<GridAgent, GameObject>();

    void Start()
    {
        foreach (GridAgent agent in activeAgents)
        {
            SpawnNewCargoForAgent(agent);
        }
    }

    public void RegisterAgent(GridAgent agent)
    {
        if (agent != null && !activeAgents.Contains(agent))
        {
            activeAgents.Add(agent);
        }
    }

    public Vector3 GridToWorld(Vector2Int gridPos)
    {
        return new Vector3(gridPos.x - 4.5f, surfaceLevelY + 0.5f, gridPos.y - 4.5f);
    }

    public Vector2Int GetCargoLocation(GridAgent agent)
    {
        if (cargoLocations.TryGetValue(agent, out Vector2Int location))
        {
            return location;
        }

        return Vector2Int.zero;
    }

    public bool HasActiveCargo(GridAgent agent)
    {
        return activeCargoInstances.ContainsKey(agent);
    }

    public void SpawnNewCargoForAgent(GridAgent agent)
    {
        if (agent == null || cargoPrefab == null || cargoSpawnLocations.Count == 0)
        {
            return;
        }

        ClearActiveCargoForAgent(agent);

        Vector2Int selectedLocation = SelectCargoLocation(agent);
        cargoLocations[agent] = selectedLocation;

        GameObject cargoInstance = Instantiate(cargoPrefab, GridToWorld(selectedLocation), Quaternion.identity);
        activeCargoInstances[agent] = cargoInstance;
    }

    public GameObject TryPickupCargo(GridAgent agent, Vector2Int agentPos)
    {
        if (!cargoLocations.TryGetValue(agent, out Vector2Int cargoLocation))
        {
            return null;
        }

        if (agentPos != cargoLocation)
        {
            return null;
        }

        if (!activeCargoInstances.TryGetValue(agent, out GameObject cargoInstance) || cargoInstance == null)
        {
            return null;
        }

        activeCargoInstances.Remove(agent);
        return cargoInstance;
    }

    public void ClearCargoForAgent(GridAgent agent)
    {
        ClearActiveCargoForAgent(agent);
        cargoLocations.Remove(agent);
    }

    public bool IsCellOccupiedByOtherAgent(Vector2Int targetPos, GridAgent self)
    {
        return GetAgentAtCell(targetPos, self) != null;
    }

    public GridAgent GetAgentAtCell(Vector2Int targetPos, GridAgent self)
    {
        foreach (GridAgent agent in activeAgents)
        {
            if (agent != null && agent != self && agent.currentGridPos == targetPos)
            {
                return agent;
            }
        }

        return null;
    }

    public void HandleAgentCollision(GridAgent initiator, GridAgent otherAgent)
    {
        foreach (GridAgent agent in activeAgents)
        {
            if (agent != null)
            {
                agent.AddReward(-globalCollisionPenalty);
            }
        }

        foreach (GridAgent agent in activeAgents)
        {
            if (agent != null)
            {
                agent.EndEpisode();
            }
        }
    }

    private Vector2Int SelectCargoLocation(GridAgent owner)
    {
        List<Vector2Int> candidates = new List<Vector2Int>();

        foreach (Vector2Int candidate in cargoSpawnLocations)
        {
            if (candidate == deliveryLocation)
            {
                continue;
            }

            if (IsCellOccupiedByAnyAgent(candidate))
            {
                continue;
            }

            if (IsCargoAlreadyAssigned(candidate, owner))
            {
                continue;
            }

            candidates.Add(candidate);
        }

        if (candidates.Count == 0)
        {
            candidates.AddRange(cargoSpawnLocations);
        }

        int randomIndex = Random.Range(0, candidates.Count);
        return candidates[randomIndex];
    }

    private bool IsCellOccupiedByAnyAgent(Vector2Int targetPos)
    {
        foreach (GridAgent agent in activeAgents)
        {
            if (agent != null && agent.currentGridPos == targetPos)
            {
                return true;
            }
        }

        return false;
    }

    private bool IsCargoAlreadyAssigned(Vector2Int targetPos, GridAgent owner)
    {
        foreach (KeyValuePair<GridAgent, Vector2Int> pair in cargoLocations)
        {
            if (pair.Key != owner && pair.Value == targetPos && activeCargoInstances.ContainsKey(pair.Key))
            {
                return true;
            }
        }

        return false;
    }

    private void ClearActiveCargoForAgent(GridAgent agent)
    {
        if (activeCargoInstances.TryGetValue(agent, out GameObject cargoInstance) && cargoInstance != null)
        {
            Destroy(cargoInstance);
        }

        activeCargoInstances.Remove(agent);
    }
}
