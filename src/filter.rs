use serde_json::Value;

pub fn matches(payload: &Value, filter: &Value) -> bool {
    if let Value::Object(map) = filter {
        for (key, condition) in map {
            // Logic Operators
            if key == "$and" {
                if let Value::Array(conditions) = condition {
                    for cond in conditions {
                        if !matches(payload, cond) { return false; }
                    }
                    continue;
                }
            }
            if key == "$or" {
                if let Value::Array(conditions) = condition {
                    let mut any = false;
                    for cond in conditions {
                        if matches(payload, cond) { 
                            any = true; 
                            break;
                        }
                    }
                    if !any { return false; }
                    continue;
                }
            }

            // Field Lookup (Dot Notation)
            let field_value = get_value_at_path(payload, key);
            
            if !check_condition(field_value, condition) {
                return false;
            }
        }
        return true;
    }
    true 
}

fn get_value_at_path<'a>(payload: &'a Value, path: &str) -> &'a Value {
    let mut current = payload;
    for key in path.split('.') {
        if let Value::Object(map) = current {
            if let Some(val) = map.get(key) {
                current = val;
            } else {
                return &Value::Null;
            }
        } else {
            return &Value::Null;
        }
    }
    current
}

fn check_condition(field: &Value, condition: &Value) -> bool {
    // 1. Direct Equality / Implicit Array Containment
    if !condition.is_object() {
        if field == condition { return true; }
        // If field is Array, check if it contains condition
        if let Value::Array(arr) = field {
            if arr.contains(condition) { return true; }
        }
        return false;
    }

    // 2. Operators
    if let Value::Object(ops) = condition {
        for (op, val) in ops {
            match op.as_str() {
                "$eq" => if field != val { return false; },
                "$ne" => if field == val { return false; },
                "$gt" => if !compare(field, val, |ordering| ordering.is_gt()) { return false; },
                "$gte" => if !compare(field, val, |ordering| ordering.is_ge()) { return false; },
                "$lt" => if !compare(field, val, |ordering| ordering.is_lt()) { return false; },
                "$lte" => if !compare(field, val, |ordering| ordering.is_le()) { return false; },
                "$in" => {
                    if let Value::Array(arr) = val {
                        // If field is array: Intersection Check
                        if let Value::Array(field_arr) = field {
                            let mut found = false;
                            for fv in field_arr {
                                if arr.contains(fv) { found = true; break; }
                            }
                            if !found { return false; }
                        } else {
                            // Scalar field: Containment Check
                            if !arr.contains(field) { return false; }
                        }
                    }
                },
                "$nin" => {
                    if let Value::Array(arr) = val {
                         // If field is array: Fail if ANY intersection
                        if let Value::Array(field_arr) = field {
                            for fv in field_arr {
                                if arr.contains(fv) { return false; }
                            }
                        } else {
                            if arr.contains(field) { return false; }
                        }
                    }
                },
                _ => {} // Ignore unknown ops
            }
        }
        return true;
    }
    
    // Fallback for non-operator object
    field == condition
}

use std::cmp::Ordering;

fn compare<F>(a: &Value, b: &Value, check: F) -> bool 
where F: Fn(Ordering) -> bool {
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => {
            if let (Some(f1), Some(f2)) = (n1.as_f64(), n2.as_f64()) {
                if let Some(ord) = f1.partial_cmp(&f2) {
                    return check(ord);
                }
            }
        },
        (Value::String(s1), Value::String(s2)) => {
            return check(s1.cmp(s2));
        },
        _ => {} 
    }
    false
}
