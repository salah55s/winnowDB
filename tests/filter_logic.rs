use serde_json::json;
use winnow_db::filter::matches;

#[test]
fn test_filter_matches() {
    let payload = json!({
        "type": "A",
        "val": 10,
        "tags": ["x", "y"],
        "nested": { "a": 1 }
    });

    // 1. Direct Equality
    assert!(matches(&payload, &json!({ "type": "A" })), "Direct equality failed");
    assert!(!matches(&payload, &json!({ "type": "B" })), "Negative equality failed");

    // 2. Comparison
    assert!(matches(&payload, &json!({ "val": { "$gt": 5 } })), "GT failed");
    assert!(!matches(&payload, &json!({ "val": { "$gt": 15 } })), "Negative GT failed");

    // 3. Array Containment (Implicit)
    assert!(matches(&payload, &json!({ "tags": "x" })), "Array containment failed");
    assert!(matches(&payload, &json!({ "tags": "y" })), "Array containment failed");
    assert!(!matches(&payload, &json!({ "tags": "z" })), "Negative array containment failed");

    // 4. Dot Notation
    assert!(matches(&payload, &json!({ "nested.a": 1 })), "Dot notation failed");

    // 5. $in Operator with Array Field (The Gap)
    // Should match if intersection is non-empty
    // x is in ["x", "z"]
    assert!(matches(&payload, &json!({ "tags": { "$in": ["x", "z"] } })), "$in Array Intersection failed");
}
