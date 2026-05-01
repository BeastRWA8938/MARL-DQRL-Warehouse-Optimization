using UnityEngine;
using System.Collections.Generic;

public class WarehouseGridManager : MonoBehaviour
{
    [Header("Grid Dimensions")]
    public int rows = 5;
    public int cols = 10;
    public float surfaceLevelY = 0f;

    [Header("Fixed Zones")]
    public Vector2Int deliveryLocation = new Vector2Int(1, 1); 

    [Header("Cargo Management")]
    public List<Vector2Int> cargoSpawnLocations = new List<Vector2Int>();
    public GameObject cargoPrefab; // Drag your sphere/cargo prefab here
    
    [HideInInspector] public Vector2Int currentCargoLocation;
    private GameObject activeCargoInstance;

    void Start()
    {
        SpawnNewCargo();
    }

    public Vector3 GridToWorld(Vector2Int gridPos)
    {
        return new Vector3(gridPos.x - 4.5f, surfaceLevelY+0.5f, gridPos.y + 0.5f);
    }

    // Spawns (or moves) the physical cargo to a new random rack
    public void SpawnNewCargo()
    {
        if (cargoSpawnLocations.Count == 0) return;

        int randomIndex = Random.Range(0, cargoSpawnLocations.Count);
        currentCargoLocation = cargoSpawnLocations[randomIndex];

        if (activeCargoInstance == null)
        {
            activeCargoInstance = Instantiate(cargoPrefab, GridToWorld(currentCargoLocation), Quaternion.identity);
        }
        else
        {
            activeCargoInstance.transform.position = GridToWorld(currentCargoLocation);
            activeCargoInstance.SetActive(true); // Ensure it is visible
        }
    }

    // Called by the agent when it successfully steps on the cargo
// Called by the agent. Returns the physical cargo so the agent can carry it.
    public GameObject GrabActiveCargo()
    {
        GameObject pickedUpCargo = activeCargoInstance;
        
        // Clear the manager's reference so it knows the rack is empty
        activeCargoInstance = null; 
        
        return pickedUpCargo;
    }
}