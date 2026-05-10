using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(LineRenderer))]
public class GridAgent : Agent
{
    private const float Gamma = 0.99f;
    private const float ShapingScale = 0.2f;
    private const float InvalidMovePenalty = 1.0f;
    private const float MinRackPenalty = 15.0f;
    private const float MinEmptyDropzonePenalty = 12.0f;
    private const float MinReverseMovePenalty = 0.75f;
    private static readonly Vector2Int[] RelativeVisionOffsets =
    {
        new Vector2Int(-1, 0),  // 1 Left
        new Vector2Int(1, 0),   // 1 Right
        new Vector2Int(0, -1),  // 1 Behind
        new Vector2Int(0, 1),   // 1 Front
        new Vector2Int(0, 2)    // 2 Front
    };

    public enum AgentPhase { SeekCargo, DeliverCargo }
    public AgentPhase currentPhase = AgentPhase.SeekCargo;

    [Header("Environment Links")]
    public WarehouseGridManager gridManager;
    public Transform holdPoint;
    private GameObject carriedCargo;

    [Header("Agent State")]
    public Vector2Int spawnGridPos = new Vector2Int(1, 0);
    public int spawnFacingDirection = 0;
    public Vector2Int currentGridPos;
    public int facingDirection; // 0=N, 1=E, 2=S, 3=W
    private bool hasCargo;

    [Header("Stats Integration")]
    public WarehouseStatsManager statsManager;
    public string agentID = "A0"; // Set to A0 for Agent 1, A1 for Agent 2 in inspector
    private int stepsSincePickup = 0;

    [Header("Reward Tuning")]
    public float rackPenalty = MinRackPenalty;
    public float emptyDropzonePenalty = MinEmptyDropzonePenalty;
    public float reverseMovePenalty = MinReverseMovePenalty;

    [Header("Visuals")]
    public float lineOffsetHeight = 0.5f;
    private LineRenderer targetLine;

    public override void Initialize()
    {
        EnforceMinimumPenalties();

        targetLine = GetComponent<LineRenderer>();
        if (targetLine != null)
        {
            targetLine.startWidth = 0.05f;
            targetLine.endWidth = 0.05f;
            targetLine.positionCount = 2;
        }
        if (statsManager != null) 
        {
            statsManager.RegisterAgent(this, agentID);
        }
    }

    public void Awake()
    {
        if (gridManager != null)
        {
            gridManager.RegisterAgent(this);
        }
    }

    public override void OnEpisodeBegin()
    {
        // 1. Reset Phase
        currentPhase = AgentPhase.SeekCargo;
        hasCargo = false;

        // 2. Destroy currently held cargo if the episode timed out or failed
        if (carriedCargo != null) 
        {
            Destroy(carriedCargo);
            carriedCargo = null;
        }
        
        // 3. Reset Position before selecting cargo so spawn cells are avoided
        currentGridPos = spawnGridPos; 
        facingDirection = spawnFacingDirection;

        // 4. Spawn this agent's next cargo
        gridManager.SpawnNewCargoForAgent(this);
        
        // 5. Update physical visuals instantly
        UpdatePhysicalPosition();
        transform.rotation = Quaternion.Euler(0, facingDirection * 90f, 0);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // OBSERVATION 1: Phase Indicator
        sensor.AddObservation(currentPhase == AgentPhase.DeliverCargo ? 1.0f : 0.0f);

        // OBSERVATION 2 & 3: Global Target Coordinates (Normalized)
        Vector2Int currentTarget = (currentPhase == AgentPhase.SeekCargo) ? gridManager.GetCargoLocation(this) : gridManager.deliveryLocation;
        sensor.AddObservation((float)currentTarget.x / gridManager.cols); 
        sensor.AddObservation((float)currentTarget.y / gridManager.rows);

        // OBSERVATION 4 & 5: Global Agent Coordinates (Normalized)
        sensor.AddObservation((float)currentGridPos.x / gridManager.cols);
        sensor.AddObservation((float)currentGridPos.y / gridManager.rows);

        // OBSERVATIONS 6 to 9: Facing Direction One-Hot
        for (int dir = 0; dir < 4; dir++)
        {
            sensor.AddObservation(facingDirection == dir ? 1.0f : 0.0f);
        }

        // OBSERVATIONS 10 to 14: Local Vision Array (5 floats)
        foreach (Vector2Int offset in RelativeVisionOffsets)
        {
            Vector2Int rotatedOffset = RotateVector(offset, facingDirection);
            Vector2Int globalVisionPos = currentGridPos + rotatedOffset;
            float tileState = 0.0f; // 0 = Empty

            // Check walls / OOB
            if (globalVisionPos.x < 0 || globalVisionPos.x >= gridManager.cols || 
                globalVisionPos.y < 0 || globalVisionPos.y >= gridManager.rows)
            {
                tileState = 1.0f; 
            }
            else if (gridManager.IsCellOccupiedByOtherAgent(globalVisionPos, this))
            {
                tileState = 4.0f; // 4 = Other Agent
            }
            // Check Cargo (Only visible if we are seeking it)
            else if (globalVisionPos == gridManager.GetCargoLocation(this) && currentPhase == AgentPhase.SeekCargo && gridManager.HasActiveCargo(this))
            {
                tileState = 2.0f; 
            }
            // Check Delivery Zone
            else if (globalVisionPos == gridManager.deliveryLocation)
            {
                tileState = 3.0f; 
            }
            // 5 = Rack / cargo spawn cell. Own active cargo still takes priority above.
            else if (IsRackLocation(globalVisionPos))
            {
                tileState = 5.0f;
            }

            sensor.AddObservation(tileState);
        }
        // Total Observations Added: 1 + 2 + 2 + 4 + 5 = 14.
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (statsManager != null)
        {
            statsManager.IncrementGlobalStep();
        }

        if (currentPhase == AgentPhase.DeliverCargo) {
            stepsSincePickup++;
        }
        int action = actions.DiscreteActions[0];
        
        Vector2Int currentTarget = (currentPhase == AgentPhase.SeekCargo) ? gridManager.GetCargoLocation(this) : gridManager.deliveryLocation;

        // PBRS Phi(s) BEFORE action
        float phiS = CalculatePotential(currentGridPos, currentTarget);
        
        float stepReward = 0f;
        bool shouldCalculatePBRS = false;
        bool phaseChangedThisStep = false;
        bool episodeEnded = false;

        // --- ACTION LOGIC ---
        // Actions 2 & 3: Rotations
        if (action == 2 || action == 3) 
        {
            RotateAgent(action == 2 ? 1 : -1);
            stepReward -= 0.05f; // Base step penalty
            shouldCalculatePBRS = false; 
        }
        // Actions 0 & 1: Movement (Forward & Backward)
        else if (action == 0 || action == 1) 
        {
            Vector2Int moveDir = GetForwardVector(facingDirection);
            
            // If Action 1 (Backward), invert the movement vector
            if (action == 1)
            {
                moveDir = new Vector2Int(-moveDir.x, -moveDir.y);
                stepReward -= reverseMovePenalty;
            }

            Vector2Int nextPos = currentGridPos + moveDir;

            // Bounds Check
            if (nextPos.x < 0 || nextPos.x >= gridManager.cols || 
                nextPos.y < 0 || nextPos.y >= gridManager.rows)
            {
                stepReward -= InvalidMovePenalty;
                shouldCalculatePBRS = false; // Do not calculate PBRS on OOB
            }
            else if (gridManager.IsCellOccupiedByOtherAgent(nextPos, this))
            {
                stepReward -= InvalidMovePenalty;
                AddReward(stepReward);
                
                // STATS LOGIC: Record teammate collision
                if (statsManager != null) statsManager.RecordCollision();
                
                gridManager.HandleAgentCollision(this, gridManager.GetAgentAtCell(nextPos, this));
                return;
            }
            else
            {
                // Valid Move Execution
                currentGridPos = nextPos;
                UpdatePhysicalPosition(); 
                
                stepReward -= 0.05f; // Base step penalty
                shouldCalculatePBRS = true;

                // --- PHASE 1 ---
                if (currentPhase == AgentPhase.SeekCargo)
                {
                    if (currentGridPos == gridManager.deliveryLocation)
                    {
                        stepReward -= emptyDropzonePenalty;
                        if (statsManager != null) statsManager.RecordEmptyDrop();
                    }

                    if (currentGridPos == gridManager.GetCargoLocation(this))
                    {
                        if (HandleVisualPickup())
                        {
                            stepReward += 15.0f; // Pickup Sparse Reward
                            currentPhase = AgentPhase.DeliverCargo;
                            stepsSincePickup = 0;
                            
                            phaseChangedThisStep = true;
                            shouldCalculatePBRS = false; 
                        }
                    }
                    else if (IsRackLocation(currentGridPos))
                    {
                        stepReward -= rackPenalty;
                        if (statsManager != null) statsManager.RecordRackHit(this);
                    }
                }
                // --- PHASE 2 ---
                else if (currentPhase == AgentPhase.DeliverCargo)
                {
                    if (currentGridPos == gridManager.deliveryLocation && hasCargo)
                    {
                        stepReward += 50.0f; // Delivery Success Sparse Reward
                        shouldCalculatePBRS = false; 
                        episodeEnded = true;
                        
                        if (statsManager != null) statsManager.RecordDelivery(this, stepsSincePickup);
                        HandleVisualDrop(); 
                    }
                    else if (IsRackLocation(currentGridPos))
                    {
                        stepReward -= rackPenalty;
                        if (statsManager != null) statsManager.RecordRackHit(this);
                    }
                }
            }
        }

        // --- APPLY SCALED PBRS ---
        // Formula: F(s, a, s') = gamma * Phi(s') - Phi(s)
        if (shouldCalculatePBRS && !phaseChangedThisStep && !episodeEnded)
        {
            float phiS_Prime = CalculatePotential(currentGridPos, currentTarget);
            float shapingReward = (Gamma * phiS_Prime) - phiS;
            stepReward += ShapingScale * shapingReward;
        }

        AddReward(stepReward);

        if (episodeEnded)
        {
            EndEpisode();
        }
    }

    void Update()
    {
        if (Application.isBatchMode)
        {
            return;
        }

        UpdateTargetAndLine();
    }

    // --- Helpers ---
    private float CalculatePotential(Vector2Int position, Vector2Int target)
    {
        // Phi(s) = -ManhattanDistance
        int distance = Mathf.Abs(position.x - target.x) + Mathf.Abs(position.y - target.y);
        return -distance;
    }

    private void RotateAgent(int direction)
    {
        facingDirection = (facingDirection + direction + 4) % 4;
        transform.rotation = Quaternion.Euler(0, facingDirection * 90f, 0);
    }

    private Vector2Int GetForwardVector(int dir)
    {
        switch (dir)
        {
            case 0: return new Vector2Int(0, 1);  
            case 1: return new Vector2Int(1, 0);  
            case 2: return new Vector2Int(0, -1); 
            case 3: return new Vector2Int(-1, 0); 
            default: return Vector2Int.zero;
        }
    }

    private Vector2Int RotateVector(Vector2Int v, int dir)
    {
        switch (dir)
        {
            case 0: return v;                               
            case 1: return new Vector2Int(v.y, -v.x);       
            case 2: return new Vector2Int(-v.x, -v.y);      
            case 3: return new Vector2Int(-v.y, v.x);       
            default: return v;
        }
    }

    private void OnValidate()
    {
        EnforceMinimumPenalties();
    }

    private void EnforceMinimumPenalties()
    {
        rackPenalty = Mathf.Max(rackPenalty, MinRackPenalty);
        emptyDropzonePenalty = Mathf.Max(emptyDropzonePenalty, MinEmptyDropzonePenalty);
        reverseMovePenalty = Mathf.Max(reverseMovePenalty, MinReverseMovePenalty);
    }

    private bool IsRackLocation(Vector2Int gridPos)
    {
        return gridManager.cargoSpawnLocations.Contains(gridPos);
    }

    private void UpdatePhysicalPosition() 
    { 
        transform.position = gridManager.GridToWorld(currentGridPos); 
    }

    private bool HandleVisualPickup() 
    { 
        carriedCargo = gridManager.TryPickupCargo(this, currentGridPos);
        if (carriedCargo != null)
        {
            hasCargo = true;
            carriedCargo.transform.SetParent(holdPoint);
            carriedCargo.transform.localPosition = Vector3.zero;
            carriedCargo.transform.localRotation = Quaternion.identity;
            return true;
        }

        return false;
    }

    private void HandleVisualDrop() 
    { 
        hasCargo = false;
        if (carriedCargo != null) 
        {
            Destroy(carriedCargo); 
            carriedCargo = null;
        }
    }

    private void UpdateTargetAndLine()
    {
        if (targetLine == null) return;
        Vector2Int currentTarget = (currentPhase == AgentPhase.SeekCargo) ? gridManager.GetCargoLocation(this) : gridManager.deliveryLocation;
        Vector3 startPos = transform.position + Vector3.up * lineOffsetHeight;
        Vector3 endPos = gridManager.GridToWorld(currentTarget) + Vector3.up * lineOffsetHeight;
        
        targetLine.SetPosition(0, startPos);
        targetLine.SetPosition(1, endPos);
        targetLine.startColor = (currentPhase == AgentPhase.DeliverCargo) ? Color.green : Color.red;
        targetLine.endColor = (currentPhase == AgentPhase.DeliverCargo) ? Color.green : Color.red;
    }
}
