import os
import gzip
import json
import argparse


def _iter_json_gz(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return


def _initial_counts(path, max_records=None):
    u, i = {}, {}
    n = 0
    for obj in _iter_json_gz(path):
        uid = obj.get("reviewerID")
        asin = obj.get("asin")
        if not uid or not asin:
            continue
        u[uid] = u.get(uid, 0) + 1
        i[asin] = i.get(asin, 0) + 1 
    return u, i


def _build_user_sequences(path, valid_users, valid_items, max_records=None, min_rating=3):
    import random
    
    seqs = {}
    used_items = set()
    filtered_count = 0
    
    # First collect all valid sequences
    for obj in _iter_json_gz(path):
        uid = obj.get("reviewerID")
        asin = obj.get("asin")
        if not uid or not asin:
            continue
        if uid not in valid_users or asin not in valid_items:
            continue
        ts = obj.get("unixReviewTime")
        rating = obj.get("overall")
        if ts is None:
            continue
        if rating is not None and float(rating) < min_rating:
            filtered_count += 1
            continue
        lst = seqs.get(uid)
        if lst is None:
            lst = []
            seqs[uid] = lst
        lst.append((int(ts), asin, float(rating) if rating is not None else None))
        used_items.add(asin)
      
    for uid in list(seqs.keys()):
        seqs[uid].sort(key=lambda x: x[0])
    
    print(f"Filtered {filtered_count} interactions with rating < {min_rating}")
    print(f"Total sequences before sampling: {len(seqs)}")
    
    # Randomly sample sequences if max_records is specified
    if max_records is not None and len(seqs) > max_records:
        sampled_users = random.sample(list(seqs.keys()), max_records)
        sampled_seqs = {uid: seqs[uid] for uid in sampled_users}
        
        sampled_used_items = set()
        for uid, user_seq in sampled_seqs.items():
            for _, asin, _ in user_seq:
                sampled_used_items.add(asin)
        
        print(f"Randomly sampled {len(sampled_seqs)} sequences from {len(seqs)} total sequences")
        return sampled_seqs, sampled_used_items
    
    # Return all sequences if no sampling was performed
    return seqs, used_items
    

def _write_user_sequences(out_path, seqs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for uid, lst in seqs.items():
            f.write(
                json.dumps(
                    {
                        "user_id": uid,
                        "items": [
                            {"asin": asin, "timestamp": ts, "rating": rating}
                            for ts, asin, rating in lst
                        ],
                        "count": len(lst),
                    }
                )
                + "\n"
            )


def _analyze_meta_fields(meta_path, sample_size=5):
    """Analyze and print meta fields from the first few items"""
    print("\n=== Meta Fields Analysis ===")
    field_counts = {}
    sample_items = []
    
    for i, obj in enumerate(_iter_json_gz(meta_path)):
        if i < sample_size:
            sample_items.append(obj)
        
        # Count field occurrences
        for field in obj.keys():
            field_counts[field] = field_counts.get(field, 0) + 1
        
        if i >= 1000:  # Stop after analyzing enough items
            break
    
    print(f"Found {len(field_counts)} unique fields in meta data:")
    for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {field}: {count} occurrences")
    
    print(f"\nSample of {len(sample_items)} meta items:")
    for i, item in enumerate(sample_items):
        print(f"\nItem {i+1}:")
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {key}: {value}")


def _analyze_review_fields(review_path, sample_size=5):
    """Analyze and print review fields from the first few reviews"""
    print("\n=== Review Fields Analysis ===")
    field_counts = {}
    sample_reviews = []
    rating_distribution = {}
    
    for i, obj in enumerate(_iter_json_gz(review_path)):
        if i < sample_size:
            sample_reviews.append(obj)
        
        # Count field occurrences
        for field in obj.keys():
            field_counts[field] = field_counts.get(field, 0) + 1
        
        # Count rating distribution
        rating = obj.get("overall")
        if rating is not None:
            rating = float(rating)
            rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
        
        if i >= 1000:  # Stop after analyzing enough items
            break
    
    print(f"Found {len(field_counts)} unique fields in review data:")
    for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {field}: {count} occurrences")
    
    print(f"\nRating distribution (first 1000 reviews):")
    for rating in sorted(rating_distribution.keys()):
        print(f"  Rating {rating}: {rating_distribution[rating]} reviews")
    
    print(f"\nSample of {len(sample_reviews)} reviews:")
    for i, review in enumerate(sample_reviews):
        print(f"\nReview {i+1}:")
        for key, value in review.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {key}: {value}")


def _collect_reviews_by_asin(review_path, keep_items):
    """Collect reviewText for each ASIN from review data"""
    reviews_by_asin = {}
    for obj in _iter_json_gz(review_path):
        asin = obj.get("asin")
        review_text = obj.get("reviewText")
        
        if asin and asin in keep_items and review_text:
            if asin not in reviews_by_asin:
                reviews_by_asin[asin] = []
            reviews_by_asin[asin].append(review_text)
    
    return reviews_by_asin


def _write_item_meta(meta_path, out_path, keep_items, reviews_by_asin=None):
    """Write item meta data with only specified fields and optional reviews"""
    keep_fields = {"category", "description", "title", "brand", "price", "asin", "imageURL"}
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        n = 0
        for obj in _iter_json_gz(meta_path):
            asin = obj.get("asin")
            if asin and asin in keep_items:
                # Filter object to only keep specified fields
                filtered_obj = {k: v for k, v in obj.items() if k in keep_fields}
                
                # Add review list if available
                if reviews_by_asin and asin in reviews_by_asin:
                    filtered_obj["reviews"] = reviews_by_asin[asin]
                
                f.write(json.dumps(filtered_obj) + "\n")


def _find_items_with_missing_fields(meta_path, keep_items):
    """Find items that have missing fields from the keep_fields set"""
    keep_fields = {"category", "description", "title", "brand", "price", "asin", "imageURL"}
    items_with_missing_fields = []
    
    for obj in _iter_json_gz(meta_path):
        asin = obj.get("asin")
        if asin and asin in keep_items:
            # Check which fields are missing
            missing_fields = []
            for field in keep_fields:
                if field not in obj or obj[field] is None or (isinstance(obj[field], str) and not obj[field].strip()):
                    missing_fields.append(field)
            
            if missing_fields:
                # Get item title if available, otherwise use asin
                item_name = obj.get("title", asin)
                items_with_missing_fields.append({
                    "asin": asin,
                    "name": item_name,
                    "missing_fields": missing_fields
                })
    
    return items_with_missing_fields


def _write_filter_stats(out_path, valid_users, valid_items, u_counts, i_counts, k):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        stats = {
            "total_users": len(u_counts),
            "total_items": len(i_counts),
            "filtered_users": len(valid_users),
            "filtered_items": len(valid_items),
            "user_threshold": k,
            "item_threshold": k,
            "user_counts": {u: c for u, c in u_counts.items() if u in valid_users},
            "item_counts": {a: c for a, c in i_counts.items() if a in valid_items}
        }
        f.write(json.dumps(stats, indent=2) + "\n")


def _sample_users(seqs, sample_size):
    """Sample a subset of users from the sequences"""
    import random
    user_ids = list(seqs.keys())
    if len(user_ids) <= sample_size:
        return seqs
    
    sampled_users = random.sample(user_ids, sample_size)
    return {uid: seqs[uid] for uid in sampled_users}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--sample_users", type=int, default=None)
    parser.add_argument("--min_rating", type=float, default=3.0)
    args = parser.parse_args()

    inp_dir = args.input_dir
    five_core = os.path.join(inp_dir, "Electronics.json.gz")
    meta_path = os.path.join(inp_dir, "meta_Electronics.json.gz")
    out_dir = os.path.join(inp_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)

    source_path = five_core
    
    # Analyze both meta and review fields
    print("Analyzing review data fields...")
    _analyze_review_fields(source_path)
    print("\nAnalyzing meta data fields...")
    _analyze_meta_fields(meta_path)
    
    u_counts, i_counts = _initial_counts(source_path)
    valid_users = {u for u, c in u_counts.items() if c >= args.k}
    valid_items = {a for a, c in i_counts.items() if c >= args.k}

    seqs, used_items = _build_user_sequences(source_path, valid_users, valid_items, max_records=args.max_records, min_rating=args.min_rating)
    
    if args.sample_users is not None:
        seqs = _sample_users(seqs, args.sample_users)
        sampled_used_items = set()
        for user_seq in seqs.values():
            for _, asin, _ in user_seq:
                sampled_used_items.add(asin)
        used_items = sampled_used_items
    
    # Collect review texts for each ASIN
    print("Collecting review texts for items...")
    reviews_by_asin = _collect_reviews_by_asin(source_path, used_items)
    print(f"Collected reviews for {len(reviews_by_asin)} items")
    
    # Find items with missing fields
    print("Finding items with missing fields...")
    items_with_missing_fields = _find_items_with_missing_fields(meta_path, used_items)
    
    if items_with_missing_fields:
        print(f"\nFound {len(items_with_missing_fields)} items with missing fields:")
        for item in items_with_missing_fields[:10]:  # Show first 10 items
            print(f"  ASIN: {item['asin']}")
            print(f"  Name: {item['name']}")
            print(f"  Missing fields: {', '.join(item['missing_fields'])}")
            print()
        
        if len(items_with_missing_fields) > 10:
            print(f"  ... and {len(items_with_missing_fields) - 10} more items")
        
        # Save detailed list to file
        missing_fields_path = os.path.join(out_dir, "items_with_missing_fields.json")
        with open(missing_fields_path, "w", encoding="utf-8") as f:
            json.dump(items_with_missing_fields, f, indent=2, ensure_ascii=False)
        print(f"Detailed list saved to: {missing_fields_path}")
    else:
        print("No items with missing fields found!")
    
    _write_user_sequences(os.path.join(out_dir, "user_sequences_5core.jsonl"), seqs)
    _write_item_meta(meta_path, os.path.join(out_dir, "item_meta_5core.jsonl"), used_items, reviews_by_asin)
    _write_filter_stats(os.path.join(out_dir, "filter_stats.json"), valid_users, valid_items, u_counts, i_counts, args.k)


if __name__ == "__main__":
    main()
